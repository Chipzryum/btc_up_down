# --- Predictor Configuration ---
# Number of previous candles to have available for the prediction.
PREV_DATA_COUNT = 1

import sys
import os
import logging

# Add the project root to the Python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the base PricePredictor class
from Models.predictor import PricePredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PreviousCandlePredictor(PricePredictor):
    """
    A predictor that bases its prediction on the direction of the previous candle.
    - If the previous candle was UP (close > open), it predicts UP.
    - If the previous candle was DOWN (close < open), it predicts DOWN.
    """
    def __init__(self):
        super().__init__()

    def reset_state(self):
        """
        Resets the internal state of the predictor, primarily the historical data.
        """
        self.historical_data = []

    def load_historical_data(self, historical_data):
        """Loads historical data provided by the backtester."""
        if historical_data:
            self.historical_data.extend(historical_data)
            # Keep only the required amount of historical data
            if len(self.historical_data) > PREV_DATA_COUNT:
                self.historical_data = self.historical_data[-PREV_DATA_COUNT:]

    def make_prediction(self):
        """
        Makes a prediction based on the last available historical candle.
        """
        # The backtester calls load_historical_data before this.
        # We need at least one historical candle to make a prediction.
        if not self.historical_data:
            return None, 0.0, "Not enough historical data to determine previous candle's direction."

        # The historical data contains the previous candle.
        previous_candle = self.historical_data[-1]
        prev_open = previous_candle['open']
        prev_close = previous_candle['close']

        if prev_close > prev_open:
            prediction = 1  # Predict opposide
            explanation = f"Previous candle was UP (O:{prev_open:.2f}, C:{prev_close:.2f}). Predicting UP."
        elif prev_close < prev_open:
            return None, 0.0, "Not predicting DOWN based on previous candle's direction."
            #prediction = 0  # Predict opposide
            explanation = f"Previous candle was DOWN (O:{prev_open:.2f}, C:{prev_close:.2f}). Predicting DOWN."
        else:
            # If the previous candle was neutral (Doji), we don't have a signal.
            return None, 0.0, f"Previous candle was neutral (O:{prev_open:.2f}, C:{prev_close:.2f}). No prediction."

        confidence = 0.95  # High confidence in this simple rule

        return prediction, confidence, explanation
