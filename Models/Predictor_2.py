# --- Predictor Configuration ---
# These settings are specific to the prediction algorithm.
# You can use them or add your own!
PREV_DATA_UNIT = 'minutes'          # Unit for historical data ('minutes' or 'hours')
PREV_DATA_COUNT = 60                # Amount of historical data required (1 hour)
VOLUME_SPIKE_WINDOW = 5             # Rolling window for volume mean
VOLUME_SPIKE_MULTIPLIER = 1.5       # Multiplier for volume spike detection
DOJI_THRESHOLD = 0.05               # Threshold for Doji pattern detection

import pandas as pd
import logging
import sys
import os
from datetime import datetime, timedelta

# Add the project root to the Python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the base PricePredictor class
from Models.predictor import PricePredictor

# Configure logging (you can ignore this if you are new)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TechnicalAnalysisPredictor(PricePredictor):
    """
    Price prediction algorithm based on classical technical analysis indicators
    over a 1-hour window.
    Extends the base PricePredictor class.
    """
    def __init__(self):
        # Call the parent class constructor
        super().__init__()
        self.historical_data = []
        self.reset_state()

    def reset_state(self):
        """Resets the state of the predictor."""
        # Trim historical data to the required size to prevent memory leaks
        if len(self.historical_data) > PREV_DATA_COUNT:
            self.historical_data = self.historical_data[-PREV_DATA_COUNT:]

    def load_historical_data(self, historical_data):
        """
        Receives a list of dictionaries, each representing a candle from the past.
        This method is called by the backtester before processing a new hour.
        """
        if historical_data:
            self.historical_data.extend(historical_data)
            # Ensure we only keep the required number of recent candles
            if len(self.historical_data) > PREV_DATA_COUNT:
                self.historical_data = self.historical_data[-PREV_DATA_COUNT:]
            logging.info(f"Loaded historical data. Total minutes stored: {len(self.historical_data)}.")

    def process_live_update(self, candle_data):
        """
        Processes a single live data update and calls the prediction logic.
        """
        # Call the specific prediction logic for this model
        return self.make_prediction(candle_data)

    def make_prediction(self, candle_data):
        """
        Analyzes the last 1 hour of data to generate a trading signal based on
        VWAP, candlestick patterns, and volume spikes.
        """
        # Add the new candle to our historical data for analysis
        self.historical_data.append(candle_data)

        # Ensure we have enough data to perform calculations
        if len(self.historical_data) < PREV_DATA_COUNT:
            # Trim data to prevent memory leak while waiting
            if len(self.historical_data) > PREV_DATA_COUNT:
                self.historical_data = self.historical_data[-PREV_DATA_COUNT:]
            return None, 0.0, f"Waiting for historical data. Have {len(self.historical_data)}, need {PREV_DATA_COUNT}."

        # Keep data to the last 60 minutes
        self.historical_data = self.historical_data[-PREV_DATA_COUNT:]

        # Create a DataFrame for easier analysis
        df = pd.DataFrame(self.historical_data)

        # --- Technical Analysis Calculations ---

        # 1. VWAP (Volume-Weighted Average Price)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        current_vwap = vwap.iloc[-1]

        # 2. Candlestick Patterns (using the last two candles)
        current_candle = df.iloc[-1]
        prior_candle = df.iloc[-2]

        # Bullish Engulfing
        is_bullish_engulfing = (current_candle['low'] > prior_candle['low'] and
                                current_candle['close'] > prior_candle['open'])

        # Doji
        candle_range = current_candle['high'] - current_candle['low']
        is_doji = False
        if candle_range > 0:
            is_doji = abs(current_candle['open'] - current_candle['close']) < (DOJI_THRESHOLD * candle_range)

        # 3. Volume Spike Confirmation
        volume_mean = df['volume'].rolling(window=VOLUME_SPIKE_WINDOW).mean()
        has_volume_spike = current_candle['volume'] > (volume_mean.iloc[-1] * VOLUME_SPIKE_MULTIPLIER)

        # --- Trading Logic ---

        current_price = current_candle['close']
        price_above_vwap = current_price > current_vwap

        # BUY Signal: Bullish Engulfing with volume confirmation and price above VWAP
        if price_above_vwap and is_bullish_engulfing and has_volume_spike:
            prediction = 1  # 1 for UP
            confidence = 0.85
            explanation = f"BUY signal: Price ({current_price:.2f}) is above VWAP ({current_vwap:.2f}), a Bullish Engulfing pattern was detected, and confirmed with a volume spike."
            return prediction, confidence, explanation

        # SELL Signal: Doji pattern suggests indecision or reversal
        if is_doji:
            prediction = 0  # 0 for DOWN
            confidence = 0.65
            explanation = f"SELL signal: Doji pattern detected, indicating market indecision or potential reversal."
            return prediction, confidence, explanation

        # HOLD Signal: No clear signal found
        prediction = None
        confidence = 0.0
        explanation = f"HOLD signal: No clear trading pattern detected. Price vs VWAP: {price_above_vwap}, Bullish Engulfing: {is_bullish_engulfing}, Doji: {is_doji}, Volume Spike: {has_volume_spike}."
        return prediction, confidence, explanation
