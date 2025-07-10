# --- Data Fetching Configuration ---
# Used by data_fetcher.py for live data retrieval from Binance
HISTORICAL_HOURS = 30               # Number of past hours of data to fetch initially
HISTORICAL_INTERVAL = '1h'          # Time interval for historical data (e.g., '1m', '5m', '1h')
LIVE_UPDATE_INTERVAL = 1            # Minutes between live price checks

# --- Backtesting Configuration ---
# Used by backtester.py to simulate trading on historical data
BACKTEST_FILE = 'Data/btc_minute_data.csv' # Data file relative to project root
BACKTEST_TRIGGER_MINUTE = 5         # The minute within an hour at which to trigger a prediction (e.g., 5 for XX:05)
REQUIRED_MINUTES = 4                # Minimum number of data points required in an hour to be considered for backtesting

# --- Model Configuration ---
# Used to specify which model file to use from the Models folder
MODEL_FILE = 'Template_Empty.py'     # Model file name in the Models folder
