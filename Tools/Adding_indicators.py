import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def add_indicators(input_file, output_file, lookback_hours=18):
    """
    Add technical indicators to BTC data and prepare it for model training.
    Preserves all original columns and appends indicators and Target.
    """
    print(f"Adding indicators to data from {input_file}...")

    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return False

    try:
        df = pd.read_csv(input_file)

        # If columns are numbered, assign correct names
        if df.columns[0] == '0':
            df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume', 'label']

        df['open_time'] = pd.to_datetime(df['open_time'])
        df = df.sort_values('open_time').reset_index(drop=True)
        print(f"Successfully loaded {len(df)} records.")

        # Add technical indicators (do not drop or reorder original columns)
        print("Calculating technical indicators...")

        df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean().round(2)
        df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean().round(2)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD_histogram'] = (ema12 - ema26).ewm(span=9, adjust=False).mean() - (ema12 - ema26)
        df['MACD_histogram'] = df['MACD_histogram'].round(6)

        # ADX (simplified)
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        df['atr_14'] = df['tr'].rolling(14).mean()
        df['plus_di_14'] = 100 * (df['plus_dm'].rolling(14).mean() / df['atr_14'])
        df['minus_di_14'] = 100 * (df['minus_dm'].rolling(14).mean() / df['atr_14'])
        df['dx'] = 100 * (abs(df['plus_di_14'] - df['minus_di_14']) / (df['plus_di_14'] + df['minus_di_14']))
        df['ADX'] = df['dx'].rolling(14).mean().round(2)

        # RSI_14
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean().replace(0, 1e-10)
        rs = avg_gain / avg_loss
        df['RSI_14'] = (100 - (100 / (1 + rs))).round(2)

        # StochRSI
        rsi_min = df['RSI_14'].rolling(14).min()
        rsi_max = df['RSI_14'].rolling(14).max()
        rsi_range = (rsi_max - rsi_min).replace(0, 1e-10)
        df['StochRSI'] = ((df['RSI_14'] - rsi_min) / rsi_range).round(4)

        # CCI_20
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(20).mean()
        mean_dev = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        df['CCI_20'] = ((tp - sma_tp) / (0.015 * mean_dev)).round(2)

        # ROC_10
        df['ROC_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100).round(4)

        # ATR_14
        df['ATR_14'] = df['tr'].rolling(14).mean().round(2)

        # BB_upper, BB_lower
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['BB_upper'] = (sma_20 + (std_20 * 2)).round(2)
        df['BB_lower'] = (sma_20 - (std_20 * 2)).round(2)

        # MFI_14
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
        pos_mf = positive_flow.rolling(14).sum()
        neg_mf = negative_flow.rolling(14).sum().replace(0, 1e-10)
        mfi = 100 - (100 / (1 + pos_mf / neg_mf))
        df['MFI_14'] = mfi.round(2)

        # OBV (vectorized, using cumsum)
        direction = np.where(df['close'] > df['close'].shift(1), 1,
                     np.where(df['close'] < df['close'].shift(1), -1, 0))
        df['OBV'] = (direction * df['volume']).cumsum().fillna(0).round(2)

        # Candle_Body_Size (absolute value)
        df['Candle_Body_Size'] = abs(df['close'] - df['open']).round(2)

        # Wick_to_Body_Ratio
        body_size = abs(df['close'] - df['open']).replace(0, 1e-10)
        df['Wick_to_Body_Ratio'] = ((df['high'] - df['low']) / body_size).round(4)

        # Target: Up/Down for next period
        df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)

        # Replace inf with nan, drop rows with nan (for indicators)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        initial_length = len(df)
        df = df.dropna()
        final_length = len(df)
        print(f"Dropped {initial_length - final_length} rows due to NaN values. {final_length} records remain.")

        # Round indicator columns
        rounding_dict = {
            'EMA_9': 2, 'EMA_21': 2, 'MACD_histogram': 6, 'ADX': 2, 'RSI_14': 2,
            'StochRSI': 4, 'CCI_20': 2, 'ROC_10': 4, 'ATR_14': 2, 'BB_upper': 2,
            'BB_lower': 2, 'MFI_14': 2, 'OBV': 2, 'Candle_Body_Size': 2, 'Wick_to_Body_Ratio': 4
        }
        for col, decimals in rounding_dict.items():
            if col in df.columns:
                df[col] = df[col].round(decimals)

        print(f"After calculating indicators, {len(df)} valid records remain.")

        # Save all original columns, then indicators, then Target
        original_cols = [col for col in ['open_time', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume', 'label'] if col in df.columns]
        indicator_cols = [col for col in rounding_dict.keys() if col in df.columns]
        columns_to_save = original_cols + indicator_cols + ['Target']
        df[columns_to_save].to_csv(output_file, index=False)

        print(f"\nSuccessfully added indicators and saved data to {output_file}")
        print(f"Number of records: {len(df)}")
        print(f"Number of features: {len(indicator_cols)}")
        print(f"Target distribution:\n{df['Target'].value_counts(normalize=True)}")

        return True

    except Exception as e:
        print(f"Error processing data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_model_dataset(input_file, lookback_hours=18, train_test_split=0.8):
    """
    Prepare a dataset for model training with lookback windows.

    Parameters:
    -----------
    input_file : str
        Path to the CSV file with indicators
    lookback_hours : int
        Number of previous periods to use for each prediction
    train_test_split : float
        Proportion of data to use for training (0.0-1.0)

    Returns:
    --------
    X_train, X_test, y_train, y_test : train/test split of the data
    """
    print(f"Creating model dataset from {input_file} with {lookback_hours} period lookback...")

    # Read the prepared data
    df = pd.read_csv(input_file)

    # Additional check for infinite values
    if df.isin([np.inf, -np.inf]).any().any():
        print("Warning: Data contains infinite values. Replacing with NaN.")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna()
        print(f"After cleaning, {len(df)} records remain.")

    # Feature columns (exclude timestamp and target)
    feature_columns = [col for col in df.columns if col not in ['open_time', 'Target']]

    # Normalize the features
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    # Prepare sequences for LSTM
    X, y = [], []
    for i in range(lookback_hours, len(df)):
        X.append(df[feature_columns].iloc[i - lookback_hours:i].values)
        y.append(df['Target'].iloc[i])

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split into train and test sets
    split_idx = int(len(X) * train_test_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Dataset prepared with shape: X={X.shape}, y={y.shape}")
    print(f"Train set: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test set: X_test={X_test.shape}, y_test={y_test.shape}")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Configuration variables
    INPUT_FILE = 'Data/btc_data_full.csv'  # Path to input data file
    OUTPUT_FILE = 'Data/btc_hourly_indicators_data.csv'  # Path to save data with indicators

    # First add indicators to the data
    if add_indicators(INPUT_FILE, OUTPUT_FILE, lookback_hours=18):
        # Then show how to create a model dataset
        print("\n" + "=" * 80)
        print("DEMONSTRATION: Creating a model dataset with lookback windows")
        print("=" * 80)
        create_model_dataset(OUTPUT_FILE, lookback_hours=18)
