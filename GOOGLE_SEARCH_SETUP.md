# Google Search API Setup

To use the news scraping functionality, you need to set up Google Search API credentials.

## Setup Steps

### 1. Get Google Search API Key
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the "Custom Search API" service
4. Go to "Credentials" and create an API key
5. Copy the API key

### 2. Get Google Search Engine ID
1. Go to [Google Programmable Search Engine](https://programmablesearchengine.google.com/)
2. Click "Create a search engine"
3. Enter any website (e.g., `google.com`) as the site to search
4. Give your search engine a name
5. Click "Create"
6. Go to "Setup" and copy the "Search engine ID"

### 3. Configure Environment Variables
Create a `.env` file in the project root with:

```bash
# Google Search API Configuration
GOOGLE_SEARCH_API_KEY=your_google_search_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_google_search_engine_id_here
```

### 4. Alternative: Update agent_config.yaml
You can also add the credentials to `agent_config.yaml`:

```yaml
web_search:
  google_search_api_key: "your_google_search_api_key_here"
  google_search_engine_id: "your_google_search_engine_id_here"
```

## Usage

Once configured, the news scraping will automatically use Google Search API to find relevant news articles about companies and rank them by source credibility.

## API Limits

- Google Custom Search API allows 100 free queries per day
- Each search returns up to 10 results
- The system caches results to minimize API calls

## Troubleshooting

If you see "Google Search API not configured" messages:
1. Check that your `.env` file exists and has the correct variables
2. Verify your API key and engine ID are correct
3. Ensure the Custom Search API is enabled in your Google Cloud project 