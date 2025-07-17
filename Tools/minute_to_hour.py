# Configuration variables
INPUT_FILE = 'Data/btc_minute_data_training.csv'  # Path to the input minute data file
OUTPUT_FILE = 'Data/btc_hourly_data_training.csv'  # Path to save the hourly data

import pandas as pd
import os


def convert_minute_to_hour(input_file, output_file):
    """
    Convert minute-level BTC data to hourly data.

    Parameters:
    -----------
    input_file : str
        Path to the input CSV file containing minute-level data
    output_file : str
        Path to save the hourly aggregated data
    """
    print(f"Converting minute data from {input_file} to hourly data...")

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return False

    try:
        # Read the CSV file
        # Expected format: timestamp, open, high, low, close, volume
        df = pd.read_csv(input_file, header=None)

        # Name the columns
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set timestamp as index
        df.set_index('timestamp', inplace=True)

        # Resample to hourly data
        hourly_data = pd.DataFrame()
        hourly_data['open'] = df['open'].resample('1H').first()
        hourly_data['high'] = df['high'].resample('1H').max()
        hourly_data['low'] = df['low'].resample('1H').min()
        hourly_data['close'] = df['close'].resample('1H').last()
        hourly_data['volume'] = df['volume'].resample('1H').sum()
        hourly_data['open'] = df['open'].resample('1h').first()
        hourly_data['high'] = df['high'].resample('1h').max()
        hourly_data['low'] = df['low'].resample('1h').min()
        hourly_data['close'] = df['close'].resample('1h').last()
        hourly_data['volume'] = df['volume'].resample('1h').sum()

        # Reset index to get timestamp as a column
        hourly_data.reset_index(inplace=True)

        # Display a sample of data for verification (first 5 records)
        if len(hourly_data) > 0:
            print("\nSample of converted hourly data (first 5 records):")
            print(hourly_data.head(5).to_string(index=False))

        # Save to CSV
        hourly_data.to_csv(output_file, index=False, header=False, date_format='%Y-%m-%d %H:%M:%S')

        print(f"Successfully converted and saved hourly data to {output_file}")
        print(f"\nSuccessfully converted and saved hourly data to {output_file}")
        print(f"Processed {len(df)} minute records into {len(hourly_data)} hourly records.")

        # Check if a specific timestamp exists for verification
        test_date = '2025-04-12 17:00:00'
        if test_date in hourly_data['timestamp'].astype(str).values:
            idx = hourly_data[hourly_data['timestamp'].astype(str) == test_date].index[0]
            print(f"\nVerification data for {test_date}:")
            print(f"Open: {hourly_data.loc[idx, 'open']}")
            print(f"High: {hourly_data.loc[idx, 'high']}")
            print(f"Low: {hourly_data.loc[idx, 'low']}")
            print(f"Close: {hourly_data.loc[idx, 'close']}")
            print(f"Volume: {hourly_data.loc[idx, 'volume']}")

        return True

    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return False


if __name__ == "__main__":
    convert_minute_to_hour(INPUT_FILE, OUTPUT_FILE)
