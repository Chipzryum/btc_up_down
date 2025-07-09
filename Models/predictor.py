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
    TEMPLATE: Create your own price prediction algorithm!
    - Edit the make_prediction() method below.
    - Use self.history for historical data if you want.
    - Return 1 for UP, 0 for DOWN.
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
        --- YOUR ALGORITHM GOES HERE! ---
        Args:
            current_price (float): The most recent price.
            timestamp (datetime): The timestamp of the current price.
        Returns:
            tuple: (prediction, confidence, explanation)
                - prediction (int): 1 for UP, 0 for DOWN.
                - confidence (float): 0.0 to 1.0 (how sure are you?)
                - explanation (str): Short reason for your prediction.
        """
        # --- EXAMPLE: Simple Alternating Prediction ---
        # Replace this with your own logic!
        prediction = self.next_prediction
        confidence = 1.0
        explanation = f"Alternating prediction: {'UP' if prediction == 1 else 'DOWN'}"

        # Alternate for next call (remove this for your own logic)
        self.next_prediction = 1 - self.next_prediction

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
