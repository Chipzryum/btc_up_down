# --- Predictor Configuration ---
# These settings are specific to the prediction algorithm.
# You can use them or add your own!

WINDOW_SIZE = 10                    # Example: Number of historical candles to analyze
CONFIDENCE_THRESHOLD = 0.65         # Example: Minimum confidence to act on a prediction
RSI_PERIOD = 14                     # Example: Period for RSI calculation
SHORT_MA = 5                        # Example: Window for a short-term moving average
LONG_MA = 30                        # Example: Window for a long-term moving average
MAX_WAIT_MINUTES = 10               # Example: How long to wait for a signal before printing a waiting message

import pandas as pd
import logging
import sys
import os

# Add the project root to the Python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging (you can ignore this if you are new)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PricePredictor:
    """
    Base class for price prediction algorithms.
    This class provides common functionality and serves as a parent for custom predictors.
    """
    def __init__(self):
        self.history = pd.DataFrame()
        self.next_prediction = 1  # Example: alternates between UP and DOWN

    def load_historical_data(self, historical_data):
        """Loads historical data. (Optional for your algorithm)"""
        if historical_data.empty:
            logging.warning("Attempted to load empty historical data.")
            self.history = pd.DataFrame()
            return
        self.history = historical_data.sort_values('timestamp').copy()
        logging.info(f"Predictor loaded {len(self.history)} historical candles.")

    def make_prediction(self, current_price, timestamp):
        """
        Base implementation - should be overridden by subclasses.
        Args:
            current_price (float): The most recent price.
            timestamp (datetime): The timestamp of the current price.
        Returns:
            tuple: (prediction, confidence, explanation)
        """
        # Default implementation - subclasses should override this
        prediction = 1  # Default to UP
        confidence = 0.5  # Medium confidence
        explanation = "Base predictor: Default prediction (UP)"

        return prediction, confidence, explanation

    def process_live_update(self, update_data, full_history_df=None):
        """
        Processes new data (live or backtest) and returns a prediction.
        You do not need to change this function.
        """
        current_price = update_data.get('price')
        timestamp = update_data.get('timestamp')

        if current_price is None or timestamp is None:
            return None, None, "Update data is missing price or timestamp."

        return self.make_prediction(current_price, timestamp)


class AlwaysUpPredictor(PricePredictor):
    """
    Simple predictor that always predicts UP with high confidence.
    This is the default model when using predictor.py directly.
    """
    def __init__(self):
        super().__init__()
    
    def make_prediction(self, current_price, timestamp):
        """
        Always predicts UP with high confidence regardless of market conditions.
        """
        prediction = 1  # Always UP
        confidence = 0.9  # High confidence
        explanation = "Always UP Predictor: Consistently predicting UP regardless of market conditions"
        
        return prediction, confidence, explanation
