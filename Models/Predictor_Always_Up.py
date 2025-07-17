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

# Import the base PricePredictor class
from Models.predictor import PricePredictor

# Configure logging (you can ignore this if you are new)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AlwaysDownPredictor(PricePredictor):
    """
    Template model that always predicts DOWN.
    This is a simple example of how to extend the base PricePredictor class.
    """
    def __init__(self):
        # Call the parent class constructor
        super().__init__()

    def make_prediction(self, current_price, timestamp):
        """
        Simple template that always predicts DOWN regardless of price or time.
        """
        prediction = 1  # Always predict UP
        confidence = 0.9  # High confidence
        explanation = "Always Down Predictor: Consistently predicting DOWN regardless of market conditions"
        
        return prediction, confidence, explanation
