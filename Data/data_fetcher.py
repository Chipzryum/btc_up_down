import requests
import pandas as pd
import time
import logging
from datetime import datetime, timedelta

# ==============================
# CONFIGURATION (MODIFY THESE)
# ==============================
SYMBOL = "BTCUSDT"  # Trading pair
START_DATE = "2017-07-08"  # Start date in YYYY-MM-DD format
END_DATE = "2025-07-26"  # End date in YYYY-MM-DD format (None for current time)
INTERVAL = "1h"  # Kline interval
MAX_DATA_POINTS = 0  # Maximum number of candles to fetch (0 for all available)
SAVE_PATH = "Data/btc_data.csv"  # Output file path
# ==============================

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BinanceDataFetcher:
    def __init__(self, symbol, interval):
        self.spot_url = "https://api.binance.com/api/v3"
        self.futures_url = "https://fapi.binance.com/fapi/v1"
        self.symbol = symbol
        self.interval = interval
        self.historical_data = pd.DataFrame()
        logging.info(f"Initialized for {symbol} @ {interval} interval")

    def _interval_to_timedelta(self):
        """Convert Binance interval string to timedelta"""
        unit = self.interval[-1]
        value = int(self.interval[:-1])

        conversions = {
            'm': timedelta(minutes=value),
            'h': timedelta(hours=value),
            'd': timedelta(days=value),
            'w': timedelta(weeks=value),
            'M': timedelta(days=30 * value)
        }
        return conversions.get(unit, timedelta(minutes=value))

    def fetch_kline_data(self, start_time, end_time):
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        url = f"{self.spot_url}/klines"
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'startTime': start_ts,
            'endTime': end_ts,
            'limit': 1000
        }
        logging.debug(f"Fetching {self.interval} data from {start_time} to {end_time}")
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not data:
                logging.warning("No data returned for time range")
                return pd.DataFrame()

            # Process kline data (excluding specified fields)
            processed_data = [{
                'open_time': datetime.fromtimestamp(kline[0] / 1000),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5]),
                'taker_buy_base_asset_volume': float(kline[9])
            } for kline in data]

            df = pd.DataFrame(processed_data)
            df.set_index('open_time', inplace=True)
            logging.info(f"Fetched {len(df)} records")
            return df
        except requests.exceptions.RequestException as e:
            logging.error(f"Spot API request failed: {e}")
            return pd.DataFrame()

    def fetch_historical_data(self):
        start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
        end_dt = datetime.strptime(END_DATE, "%Y-%m-%d") if END_DATE else datetime.now()

        if MAX_DATA_POINTS and MAX_DATA_POINTS > 0:
            max_duration = self._interval_to_timedelta() * MAX_DATA_POINTS
            adjusted_start = end_dt - max_duration
            start_dt = max(start_dt, adjusted_start)
            logging.info(f"Adjusted start to {start_dt} based on max data points")

        current_time = start_dt
        all_data = pd.DataFrame()
        interval_delta = self._interval_to_timedelta()
        chunk_size = 1000 * interval_delta

        logging.info(f"Fetching OHLCV data from {start_dt} to {end_dt}")
        while current_time < end_dt:
            chunk_end = min(current_time + chunk_size, end_dt)

            df = self.fetch_kline_data(current_time, chunk_end)
            if df.empty:
                break

            all_data = pd.concat([all_data, df])

            if MAX_DATA_POINTS and MAX_DATA_POINTS > 0 and len(all_data) >= MAX_DATA_POINTS:
                all_data = all_data.iloc[:MAX_DATA_POINTS]
                break

            current_time = df.index[-1] + interval_delta
            time.sleep(0.1)

        if all_data.empty:
            logging.error("No OHLCV data fetched")
            return pd.DataFrame()

        # Add labels: 1 for green candle (close > open), 0 for red
        all_data['label'] = (all_data['close'] > all_data['open']).astype(int)

        # Round columns to reasonable decimal places
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in all_data.columns:
                all_data[col] = all_data[col].round(2)

        if 'volume' in all_data.columns:
            all_data['volume'] = all_data['volume'].round(4)

        if 'taker_buy_base_asset_volume' in all_data.columns:
            all_data['taker_buy_base_asset_volume'] = all_data['taker_buy_base_asset_volume'].round(4)

        self.historical_data = all_data
        logging.info(f"Total collected: {len(all_data)} records")
        return all_data.copy()

    def save_to_csv(self, filepath=SAVE_PATH):
        self.historical_data.to_csv(filepath)
        logging.info(f"Data saved to {filepath}")


if __name__ == "__main__":
    fetcher = BinanceDataFetcher(
        symbol=SYMBOL,
        interval=INTERVAL
    )

    data = fetcher.fetch_historical_data()
    if not data.empty:
        fetcher.save_to_csv()
        print(f"First 5 records:\n{data.head()}")
        print(f"Last 5 records:\n{data.tail()}")
        print(f"Data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
    else:
        logging.error("No data was fetched")
