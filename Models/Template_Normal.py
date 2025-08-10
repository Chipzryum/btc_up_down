import sys
import os
import logging
import numpy as np
import pandas as pd

# Add the project root to the Python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import the base PricePredictor class
from Models.predictor import PricePredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedPredictor(PricePredictor):
    """
    A predictor that uses technical indicators to make predictions and focuses on minimizing loss streaks.
    """
    def __init__(self, opposite=False):
        super().__init__()
        self.opposite = opposite
        self.historical_data = []
        self.PREV_DATA_COUNT = 20  # Number of previous candles to consider
        self.loss_streak = 0
        self.max_loss_streak = 3  # Maximum allowed loss streak before changing strategy

    def reset_state(self):
        """
        Resets the internal state of the predictor, primarily the historical data and loss streak.
        """
        self.historical_data = []
        self.loss_streak = 0

    def load_historical_data(self, historical_data):
        """
        Loads historical data provided by the backtester.
        """
        self.historical_data = historical_data
        # Keep only the required amount of historical data
        if len(self.historical_data) > self.PREV_DATA_COUNT:
            self.historical_data = self.historical_data[-self.PREV_DATA_COUNT:]

    def calculate_technical_indicators(self, data):
        """
        Calculate technical indicators for the given historical data.
        """
        df = pd.DataFrame(data)
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['RSI'] = self.calculate_rsi(df['close'])
        df['MACD'], df['Signal_Line'] = self.calculate_macd(df['close'])
        return df

    def calculate_rsi(self, prices, period=14):
        """
        Calculate the Relative Strength Index (RSI) for the given price data.
        """
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1.+rs)

        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up*(period-1) + upval)/period
            down = (down*(period-1) + downval)/period
            rs = up/down
            rsi[i] = 100. - 100./(1.+rs)

        return rsi

    def calculate_macd(self, prices, slow=26, fast=12, signal=9):
        """
        Calculate the Moving Average Convergence Divergence (MACD) for the given price data.
        """
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def make_prediction(self):
        """
        Makes a prediction based on the loaded historical data.
        """
        if not self.historical_data or len(self.historical_data) < self.PREV_DATA_COUNT:
            return None, 0.0, "Not enough historical data to make a prediction."

        # Calculate technical indicators on the available historical data
        df = self.calculate_technical_indicators(self.historical_data)

        # Drop NaN values which can occur from rolling calculations
        df.dropna(inplace=True)
        if df.empty:
            return None, 0.0, "Not enough data after indicator calculation."

        # Get the latest indicators from the last row of data
        latest_data = df.iloc[-1]
        sma_5 = latest_data['SMA_5']
        sma_10 = latest_data['SMA_10']
        rsi = latest_data['RSI']
        macd = latest_data['MACD']
        signal_line = latest_data['Signal_Line']

        # Implement prediction logic based on indicators
        base_prediction, confidence, explanation = self.predict_based_on_indicators(sma_5, sma_10, rsi, macd, signal_line)

        if base_prediction is None:
            return base_prediction, confidence, explanation

        # Invert prediction if opposite is set
        if self.opposite:
            prediction = 1 - base_prediction
            explanation = f"INVERTED: {explanation}"
        else:
            prediction = base_prediction

        return prediction, confidence, explanation

    def predict_based_on_indicators(self, sma_5, sma_10, rsi, macd, signal_line):
        """
        Makes a prediction based on technical indicators.
        Prediction: 1 for UP, 0 for DOWN.
        """
        up_signals = 0
        down_signals = 0
        explanation = ""

        # SMA Crossover Strategy
        if sma_5 > sma_10:
            up_signals += 1
            explanation += "SMA_5 > SMA_10 (Up). "
        elif sma_5 < sma_10:
            down_signals += 1
            explanation += "SMA_5 < SMA_10 (Down). "

        # RSI Strategy
        if rsi < 30:
            up_signals += 1
            explanation += "RSI < 30 (Oversold, Up). "
        elif rsi > 70:
            down_signals += 1
            explanation += "RSI > 70 (Overbought, Down). "

        # MACD Strategy
        if macd > signal_line:
            up_signals += 1
            explanation += "MACD > Signal Line (Up). "
        elif macd < signal_line:
            down_signals += 1
            explanation += "MACD < Signal Line (Down). "

        # Determine final prediction
        if up_signals > down_signals:
            prediction = 1  # Predict Up
            confidence = up_signals / (up_signals + down_signals)
        elif down_signals > up_signals:
            prediction = 0  # Predict Down
            confidence = down_signals / (up_signals + down_signals)
        else:
            # If signals are tied or there are no signals, no prediction is made.
            return None, 0.0, "No clear signal from indicators."

        return prediction, confidence, explanation.strip()

# Example usage
if __name__ == "__main__":
    # Generate sample historical data to test the predictor
    np.random.seed(42)
    closes = 50000 + np.random.randn(100).cumsum() * 10
    historical_data = [
        {'open': c, 'close': c + np.random.randn()*5, 'high': c + 10, 'low': c - 10}
        for c in closes
    ]

    # --- Test with standard logic ---
    print("--- Standard Predictor ---")
    predictor = EnhancedPredictor(opposite=False)
    # We need to simulate the backtester loop, feeding data one candle at a time
    for i in range(predictor.PREV_DATA_COUNT, len(historical_data)):
        current_data_slice = historical_data[:i]
        predictor.load_historical_data(current_data_slice)
        prediction, confidence, explanation = predictor.make_prediction()

    # Print the very last prediction
    print(f"Prediction: {'UP' if prediction == 1 else 'DOWN' if prediction == 0 else 'None'}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Explanation: {explanation}\n")


    # --- Test with opposite logic ---
    print("--- Opposite Predictor ---")
    predictor_opposite = EnhancedPredictor(opposite=True)
    # Re-run the simulation for the opposite predictor
    for i in range(predictor_opposite.PREV_DATA_COUNT, len(historical_data)):
        current_data_slice = historical_data[:i]
        predictor_opposite.load_historical_data(current_data_slice)
        prediction, confidence, explanation = predictor_opposite.make_prediction()

    # Print the very last prediction
    print(f"Prediction: {'UP' if prediction == 1 else 'DOWN' if prediction == 0 else 'None'}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Explanation: {explanation}")
