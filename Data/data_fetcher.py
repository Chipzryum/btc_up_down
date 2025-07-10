import requests
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from config.config import HISTORICAL_HOURS, HISTORICAL_INTERVAL, LIVE_UPDATE_INTERVAL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BinanceDataFetcher:
    """
    Handles fetching historical and live cryptocurrency data from the Binance API.
    """
    def __init__(self, symbol="BTCUSDT"):
        """
        Initializes the data fetcher for a specific symbol.
        """
        self.base_url = "https://api.binance.com/api/v3"
        self.symbol = symbol
        self.historical_data = pd.DataFrame()
        logging.info(f"BinanceDataFetcher initialized for symbol: {self.symbol}")

    def fetch_kline_data(self, start_time, end_time, interval='1m'):
        """
        Fetches Kline (candlestick) data for a specific time range and interval.

        Args:
            start_time (datetime): The start of the time range.
            end_time (datetime): The end of the time range.
            interval (str): The candle interval (e.g., '1m', '1h').

        Returns:
            pd.DataFrame: A DataFrame containing the kline data, or an empty DataFrame on error.
        """
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)

        url = f"{self.base_url}/klines"
        params = {
            'symbol': self.symbol,
            'interval': interval,
            'startTime': start_ts,
            'endTime': end_ts,
            'limit': 1000  # Max limit per request
        }

        logging.debug(f"Fetching klines from {start_time} to {end_time} with interval {interval}.")

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
            data = response.json()

            if not data:
                logging.warning("No data returned from Binance for the specified range.")
                return pd.DataFrame()

            # Process data into a structured DataFrame
            processed_data = [{
                'timestamp': datetime.fromtimestamp(kline[0] / 1000),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5])
            } for kline in data]

            df = pd.DataFrame(processed_data)
            df.set_index('timestamp', inplace=True)
            logging.info(f"Successfully fetched {len(df)} klines.")
            return df

        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching kline data from Binance: {e}")
            return pd.DataFrame()

    def fetch_past_data(self):
        """
        Fetches historical data based on the settings in config.py.
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=HISTORICAL_HOURS)

        logging.info(f"Fetching historical data for the last {HISTORICAL_HOURS} hours.")

        df = self.fetch_kline_data(start_time, end_time, HISTORICAL_INTERVAL)
        if not df.empty:
            self.historical_data = df
            logging.info(f"Stored {len(df)} historical {HISTORICAL_INTERVAL} candles.")
        else:
            logging.warning("Failed to fetch historical data; DataFrame is empty.")

        return df.copy() # Return a copy to prevent mutation

    def get_latest_kline(self):
        """
        Gets the most recent completed kline for the symbol.
        """
        # Fetch the last 2 1-minute klines. The second to last one is guaranteed to be complete.
        url = f"{self.base_url}/klines"
        params = {'symbol': self.symbol, 'interval': '1m', 'limit': 2}

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if not data:
                logging.warning("No kline data returned from Binance.")
                return None

            # The most recent kline might be incomplete, so we take the one before it.
            # However, for live trading, we often want the current price action.
            # Let's use the most recent one, assuming the user wants up-to-the-second data.
            last_kline = data[-1]

            kline_data = {
                'timestamp': datetime.fromtimestamp(last_kline[0] / 1000),
                'open': float(last_kline[1]),
                'high': float(last_kline[2]),
                'low': float(last_kline[3]),
                'close': float(last_kline[4]),
                'volume': float(last_kline[5])
            }
            logging.debug(f"Latest kline for {self.symbol}: {kline_data}")
            return kline_data
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not fetch latest kline: {e}")
            return None
        except (KeyError, ValueError, IndexError) as e:
            logging.error(f"Error parsing kline data: {e}")
            return None

    def start_live_updates(self, callback):
        """
        Starts a loop to fetch live data and triggers a callback function.
        """
        logging.info("Starting live price updates. Press Ctrl+C to stop.")

        try:
            while True:
                kline_data = self.get_latest_kline()
                if kline_data is None:
                    logging.warning("Skipping update cycle due to kline fetch failure.")
                    time.sleep(20) # Wait longer if API fails
                    continue

                logging.debug(f"Live update: Open={kline_data['open']:.2f}, Time={kline_data['timestamp'].strftime('%H:%M:%S')}")

                # Trigger the callback with the latest kline data
                callback(kline_data)

                # Sleep until the next interval
                # We align the sleep to the start of the next minute to be in sync with candle closes
                now = datetime.now()
                seconds_to_next_minute = 60 - now.second
                time.sleep(seconds_to_next_minute)

        except KeyboardInterrupt:
            logging.info("Live updates stopped by user.")
        except Exception as e:
            logging.error(f"An unexpected error occurred during live updates: {e}", exc_info=True)
