# Competitive Analysis & Stock Prediction Agent

This project provides an interactive command-line tool for competitive analysis and stock price prediction using both traditional and deep learning models. It fetches competitor data, analyzes historical stock prices, and predicts future prices for multiple companies using yfinance, linear regression, and an optimized LSTM model (PyTorch).

## Features
- Fetch competitors for a given company using LLMs (OpenAI, Anthropic, or AWS Bedrock)
- Download historical stock data for competitors via yfinance
- Analyze and plot historical and predicted stock prices
- Predict future prices using linear regression or LSTM (PyTorch)
- Export results to CSV
- Interactive menu for all operations

## Setup
1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure your API keys**
   - Edit `agent_config.yaml` and add your API keys for OpenAI, Anthropic, or AWS Bedrock in the `llm` section (see below for format).

## agent_config.yaml Format
```yaml
redis:
  host: "localhost"
  port: 6379
  db: 0
  username: null
  password: null

llm:
  preferred_provider: "anthropic"  # openai, anthropic, or bedrock
  temperature: 0.7
  openai_api_key: "<YOUR_OPENAI_API_KEY>"
  anthropic_api_key: "<YOUR_ANTHROPIC_API_KEY>"
  aws_access_key_id: "<YOUR_AWS_ACCESS_KEY_ID>"
  aws_secret_access_key: "<YOUR_AWS_SECRET_ACCESS_KEY>"
  aws_region: "us-east-1"

agent_capabilities:
  enable_knowledge_base: true
  enable_web_search: true
  enable_code_execution: true
  enable_file_processing: true
  enable_web_ingestion: true
  enable_api_calls: true
  enable_web_scraping: true
  enable_proxy_mode: true
  enable_media_editor: true
  enable_youtube_download: true

moderator:
  default_enabled_agents:
    - knowledge_base
    - web_search
    - assistant
    - media_editor
    - youtube_download
    - code_executor
    - web_scraper

web_search:
  brave_api_key: ""
  avesapi_api_key: ""

web_scraping:
  proxy_enabled: false
  proxy_config:
    http_proxy: ""
  default_headers:
    User-Agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
  timeout: 60
  max_links_per_page: 100

knowledge_base:
  qdrant_url: "http://localhost:6333"
  qdrant_api_key: ""
  chunk_size: 1024
  chunk_overlap: 20
  similarity_top_k: 5

media_editor:
  docker_image: "sgosain/amb-ubuntu-python-public-pod"
  input_dir: "./media_input"
  output_dir: "./media_output"
  timeout: 300
  memory_limit: "2g"

youtube_download:
  docker_image: "sgosain/amb-ubuntu-python-public-pod"
  download_dir: "./youtube_downloads"
  timeout: 600
  memory_limit: "1g"
  default_audio_only: true

docker:
  timeout: 60
  memory_limit: "512m"
  images: ["sgosain/amb-ubuntu-python-public-pod"]

service:
  max_sessions: 100
  session_timeout: 3600
  log_level: "INFO"
  log_to_file: false

memory_management:
  compression:
    enabled: true
    algorithm: "lz4"
  cache:
    enabled: true
    max_size: 1000
    ttl_seconds: 300
```

## Usage
Run the main script and follow the interactive menu:
```bash
python main.py
```

- **Option 1:** Fetch competitors and download their historical stock data
- **Option 2:** Plot historical and predicted closing prices
- **Option 3:** Predict future prices for a ticker (simple trend)
- **Option 4:** Joint linear regression forecast for all companies
- **Option 5:** Predictive LSTM model for all companies
- **Option 6:** Exit

## Output
- Historical and predicted data are saved as CSVs in the `yfinance_data/` folder.
- Plots are displayed for visual analysis.

### CSV Output Details
- For each ticker, a file named `<TICKER>_historical_data.csv` is created containing historical and predicted prices.
- The CSV includes columns for date, open, and close prices. For predictions, future dates and predicted close prices are appended.
- The `compare_all_companies` and joint forecast features generate combined CSVs with columns:
  - `Date`: The date of the record (historical or predicted)
  - `Ticker`: The stock ticker symbol
  - `Close`: The actual or predicted closing price
  - `Type`: 'Historical' or 'Predicted'
  - `Current_Close`: The last known close price before prediction (for predicted rows)
  - `Growth_%`: The percent growth from the last known close to the predicted value
  - `Market_Avg`: The average close price across all companies for each date

### Plot Output Details
- Plots show both historical and predicted close prices for each ticker.
- Linear regression and LSTM predictions are shown as dashed lines, with actual data as solid lines.
- Market average and individual company trends can be visualized for comparison.

### Example Analysis
- You can visually compare the accuracy of linear regression vs. LSTM predictions.
- Growth percentages and market averages help identify outperforming companies.
- All results are suitable for further analysis in Excel, Python, or other tools.

## Notes
- API keys must be provided in `agent_config.yaml`.
- The LSTM model is optimized for best results; linear regression is also available.
- No knowledge base operations or `.env` file details are included in this README.

---

**For any issues or contributions, please open an issue or pull request.**
