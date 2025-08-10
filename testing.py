import pandas as pd
import numpy as np
from ta.momentum import StochasticOscillator, RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator

# Configuration
TOP_SENTIMENT_FEATURES = [
    'twitter_fomo', 'twitter_bearish', 'reddit_bearish', 'twitter_dip',
    'twitter_market_manipulation', 'bitcointalk_dip',
    'twitter_selling', 'bitcointalk_whales', 'reddit_market_manipulation'
]

TECHNICAL_CONFIG = {
    'stoch_window': 14,
    'stoch_smooth': 3,
    'rsi_window': 14,
    'ema_window': 20,
    'bb_window': 20,
    'bb_std': 2,
    'sentiment_ma_windows': [3, 6]  # Moving average windows for sentiment
}


def create_enhanced_dataset(ohlcv_path, sentiment_path, output_path):
    """
    Creates enhanced dataset with:
    - Technical indicators (calculated from historic data only)
    - Sentiment features with moving averages
    - Time-series sequences with lagged values
    - Derived features (no future leakage)
    """
    # 1. Load and merge datasets
    ohlcv = pd.read_csv(ohlcv_path, parse_dates=['open_time'])
    sentiment = pd.read_csv(sentiment_path, parse_dates=['date'])

    # Keep only relevant sentiment features
    sentiment = sentiment[['date'] + TOP_SENTIMENT_FEATURES]

    # Align datasets using inner join
    merged = pd.merge(
        left=ohlcv,
        right=sentiment,
        left_on='open_time',
        right_on='date',
        how='inner'
    ).drop(columns='date').sort_values('open_time').reset_index(drop=True)

    # Save original label and remove it temporarily
    label = merged['label'].copy()
    merged = merged.drop(columns=['label'])

    # 2. Calculate technical indicators (using only historic data)
    merged = calculate_technical_indicators(merged, TECHNICAL_CONFIG)

    # 3. Add derived features
    merged = add_derived_features(merged)

    # 4. Add sentiment moving averages
    for feature in TOP_SENTIMENT_FEATURES:
        for window in TECHNICAL_CONFIG['sentiment_ma_windows']:
            # Use shift to prevent lookahead bias
            merged[f'{feature}_ma{window}'] = merged[feature].rolling(window).mean().shift(1)

    # 5. Create time-shifted target (predict next hour)
    merged['next_hour_label'] = label.shift(-1)

    # Save body_ratio and remove it temporarily
    body_ratio = merged['body_ratio'].copy()
    merged = merged.drop(columns=['body_ratio'])

    # 6. Add label and body_ratio back at the end
    merged['body_ratio'] = body_ratio
    merged['label'] = label

    # 7. Remove rows with NaN values (from indicator calculations and shifting)
    merged = merged.dropna().reset_index(drop=True)

    # 7.1 Replace inf/-inf with NaN and drop them (fix for infinite values)
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    # 8. Calculate price_momentum_1h and price_momentum_4h
    merged['price_momentum_1h'] = merged['close'].pct_change(periods=1).shift(1)
    merged['price_momentum_4h'] = merged['close'].pct_change(periods=4).shift(1)

    # 9. Select only the specified features, plus ohlcv columns, open_time and next_hour_label
    ohlcv_cols = [
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume'
    ]
    selected_features = ohlcv_cols + [
        'body_ratio',
        'price_momentum_1h',
        'price_momentum_4h',
        'return_4h',
        'bb_position',
        'stoch_diff',
        'return_1h',
        'stoch_k',
        'twitter_bearish_ma6',
        'rsi_change',
        'volume_change',
        'rsi',
        'stoch_d',
        'volume_ma',
        'volume',
        'taker_buy_base_asset_volume',
        'bb_width',
        'twitter_bearish_ma3',
        'reddit_bearish_ma6',
        'bb_upper',
        'ema',                       # Extra indicator
        'next_hour_label',           # needed for sequence creation
        'label'
    ]

    # Remove duplicates in selected_features (volume, taker_buy_base_asset_volume appear twice)
    seen = set()
    selected_features = [x for x in selected_features if not (x in seen or seen.add(x))]

    # Ensure columns exist (if not, fill with NaN)
    for col in selected_features:
        if col not in merged.columns:
            merged[col] = np.nan

    merged = merged[selected_features].dropna().reset_index(drop=True)

    # 10. Save enhanced dataset
    merged.to_csv(output_path, index=False)
    print(f"✅ Created enhanced dataset with {len(merged)} records")
    print(f"⏰ Time range: {merged['open_time'].iloc[0]} to {merged['open_time'].iloc[-1]}")
    print(f"📊 Features: {len(merged.columns)} columns")
    return merged


def calculate_technical_indicators(df, config):
    """Calculate technical indicators using only historic data"""
    # Price-based features
    df['body_ratio'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)

    # Momentum indicators - using shift(1) to ensure we're only using data that would be available
    # at the time of prediction (i.e., data from previous candles)
    stoch = StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=config['stoch_window'],
        smooth_window=config['stoch_smooth']
    )
    df['stoch_k'] = stoch.stoch().shift(1)
    df['stoch_d'] = stoch.stoch_signal().shift(1)

    rsi = RSIIndicator(close=df['close'], window=config['rsi_window'])
    df['rsi'] = rsi.rsi().shift(1)

    # Trend indicators
    ema = EMAIndicator(close=df['close'], window=config['ema_window'])
    df['ema'] = ema.ema_indicator().shift(1)

    # Volatility indicators
    bb = BollingerBands(
        close=df['close'],
        window=config['bb_window'],
        window_dev=config['bb_std']
    )
    df['bb_upper'] = bb.bollinger_hband().shift(1)
    df['bb_middle'] = bb.bollinger_mavg().shift(1)
    df['bb_lower'] = bb.bollinger_lband().shift(1)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']).shift(1)

    return df


def add_derived_features(df):
    """Create derived features from existing columns"""
    # Momentum derivatives - indicators are already shifted, so no need to shift again
    df['stoch_diff'] = df['stoch_k'] - df['stoch_d']
    df['rsi_change'] = df['rsi'].diff()

    # Bollinger Band position - use current close with shifted bollinger bands
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    # Volume trends - shift to ensure we only use past data
    df['volume_change'] = df['volume'].pct_change().shift(1)
    df['volume_ma'] = df['volume'].rolling(5).mean().shift(1)

    # Price trends - shift to ensure we only use past data
    df['return_1h'] = df['close'].pct_change().shift(1)
    df['return_4h'] = df['close'].pct_change(4).shift(1)

    return df


def create_sequences(df, seq_length=24):
    """
    Create time-series sequences for LSTM training
    Returns: X (samples, timesteps, features), y (labels)
    """
    features = df.columns.difference(['open_time', 'label', 'next_hour_label'])
    X, y = [], []

    for i in range(seq_length, len(df)):
        # Features from previous 'seq_length' hours
        sequence = df[features].iloc[i - seq_length:i].values
        # Predict next hour's label
        target = df['next_hour_label'].iloc[i]

        X.append(sequence)
        y.append(target)

    return np.array(X), np.array(y)


# Example usage
if __name__ == "__main__":
    # Update these paths
    ohlcv_file = "Data/Hourly/btc_data_full.csv"
    sentiment_file = "Data/augmento_btc.csv"
    output_file = "enhanced_btc_dataset.csv"

    # Create enhanced dataset
    enhanced_df = create_enhanced_dataset(ohlcv_file, sentiment_file, output_file)

    # Create sequences for LSTM training (example)
    print("\nCreating time-series sequences...")
    X, y = create_sequences(enhanced_df, seq_length=24)
    print(f"Sequences shape: {X.shape} (samples, timesteps, features)")
    print(f"Labels shape: {y.shape}")
