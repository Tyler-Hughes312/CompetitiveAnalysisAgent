import redis
import json
import pickle
import hashlib
from datetime import datetime, timedelta
import yaml
import os
from typing import Any, Optional, Union
import pandas as pd

class RedisCache:
    def __init__(self, config_path="agent_config.yaml"):
        """Initialize Redis cache with configuration from agent_config.yaml"""
        self.redis_client = None
        self.config = self._load_config(config_path)
        self._connect_redis()
    
    def _load_config(self, config_path: str) -> dict:
        """Load Redis configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('redis', {})
        except Exception as e:
            print(f"Warning: Could not load Redis config from {config_path}: {e}")
            return {
                'host': 'localhost',
                'port': 6379,
                'db': 0,
                'username': None,
                'password': None
            }
    
    def _connect_redis(self):
        """Establish Redis connection with error handling"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 6379),
                db=self.config.get('db', 0),
                username=self.config.get('username'),
                password=self.config.get('password'),
                decode_responses=False,  # Keep as bytes for pickle compatibility
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            print("Redis cache connected successfully")
        except Exception as e:
            print(f"Warning: Redis connection failed: {e}")
            self.redis_client = None
    
    def _generate_key(self, prefix: str, *args) -> str:
        """Generate a unique cache key from prefix and arguments"""
        # Create a hash of the arguments
        args_str = json.dumps(args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode()).hexdigest()
        return f"{prefix}:{args_hash}"
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for Redis storage"""
        try:
            # Try pickle first (handles most Python objects including DataFrames)
            return pickle.dumps(data)
        except Exception:
            # Fallback to JSON for simple types
            return json.dumps(data).encode('utf-8')
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data from Redis storage"""
        try:
            # Try pickle first
            return pickle.loads(data)
        except Exception:
            # Fallback to JSON
            return json.loads(data.decode('utf-8'))
    
    def get(self, prefix: str, *args, default=None) -> Optional[Any]:
        """Get cached data by prefix and arguments"""
        if not self.redis_client:
            return default
        
        try:
            key = self._generate_key(prefix, *args)
            cached_data = self.redis_client.get(key)
            if cached_data is not None:
                print(f"Cache HIT for {prefix}")
                return self._deserialize_data(cached_data)
            else:
                print(f"Cache MISS for {prefix}")
                return default
        except Exception as e:
            print(f"Error retrieving from cache: {e}")
            return default
    
    def set(self, prefix: str, data: Any, *args, ttl_seconds: int = 3600) -> bool:
        """Set cached data with TTL (default 1 hour)"""
        if not self.redis_client:
            return False
        
        try:
            key = self._generate_key(prefix, *args)
            serialized_data = self._serialize_data(data)
            self.redis_client.setex(key, ttl_seconds, serialized_data)
            print(f"Cached data for {prefix} with TTL {ttl_seconds}s")
            return True
        except Exception as e:
            print(f"Error setting cache: {e}")
            return False
    
    def delete(self, prefix: str, *args) -> bool:
        """Delete cached data by prefix and arguments"""
        if not self.redis_client:
            return False
        
        try:
            key = self._generate_key(prefix, *args)
            result = self.redis_client.delete(key)
            if result:
                print(f"Deleted cache for {prefix}")
            return bool(result)
        except Exception as e:
            print(f"Error deleting from cache: {e}")
            return False
    
    def clear_all(self, prefix: str = None) -> bool:
        """Clear all cached data or data with specific prefix"""
        if not self.redis_client:
            return False
        
        try:
            if prefix:
                pattern = f"{prefix}:*"
                keys = self.redis_client.keys(pattern)
            else:
                keys = self.redis_client.keys("*")
            
            if keys and len(keys) > 0:
                self.redis_client.delete(*keys)
                print(f"Cleared {len(keys)} cache entries")
            return True
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False
    
    def exists(self, prefix: str, *args) -> bool:
        """Check if cached data exists"""
        if not self.redis_client:
            return False
        
        try:
            key = self._generate_key(prefix, *args)
            result = self.redis_client.exists(key)
            return bool(result) if result is not None else False
        except Exception as e:
            print(f"Error checking cache existence: {e}")
            return False
    
    def get_ttl(self, prefix: str, *args) -> int:
        """Get remaining TTL for cached data in seconds"""
        if not self.redis_client:
            return -1
        
        try:
            key = self._generate_key(prefix, *args)
            ttl = self.redis_client.ttl(key)
            return int(ttl) if ttl is not None else -1
        except Exception as e:
            print(f"Error getting TTL: {e}")
            return -1

# Global cache instance
_cache_instance = None

def get_cache() -> RedisCache:
    """Get or create global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance
