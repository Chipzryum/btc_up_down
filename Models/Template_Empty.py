# --- Predictor Configuration ---
# These settings are specific to the prediction algorithm.
# You can use them or add your own!
PREV_DATA_UNIT = 'minutes'          # Unit for historical data ('minutes' or 'hours')
PREV_DATA_COUNT = 60                # Amount of historical data required
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
from datetime import datetime, timedelta

# Add the project root to the Python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the base PricePredictor class
from Models.predictor import PricePredictor

# Configure logging (you can ignore this if you are new)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FiveMinutePrediction(PricePredictor):
    """
    Price prediction algorithm based on price and volume change over the first 5 minutes of an hour.
    Extends the base PricePredictor class.
    """
    def __init__(self):
        # Call the parent class constructor
        super().__init__()
        self.historical_data = []
        self.reset_state()

    def reset_state(self):
        """Resets the state of the predictor for a new hour."""
        # Variables to store the start time, price, and volume of the hour
        self.hour_start_time = None
        self.start_of_hour_price = None
        self.start_of_hour_volume = None
        self.last_prediction = None
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
        Waits for 5 minutes after the hour starts and predicts based on price and volume movement.
        """
        # Example check for historical data
        if len(self.historical_data) < PREV_DATA_COUNT:
            return None, 0.0, f"Waiting for historical data. Have {len(self.historical_data)}, need {PREV_DATA_COUNT}."

        current_price = candle_data['open']
        current_volume = candle_data['volume']
        timestamp = candle_data['timestamp']

        # If starting a new hour
        if self.hour_start_time is None or timestamp.hour != self.hour_start_time.hour:
            self.hour_start_time = timestamp.replace(minute=0, second=0, microsecond=0)
            self.start_of_hour_price = current_price
            self.start_of_hour_volume = current_volume
            self.last_prediction = None
            return None, 0.0, "Recording start of hour price and volume."

        # Check if we've passed the 5-minute mark after the hour starts
        if timestamp >= self.hour_start_time + timedelta(minutes=5) and self.last_prediction is None:
            # --- New Prediction Logic ---

            # 1. Determine previous hour's trend from historical data
            prev_hour_open = self.historical_data[0]['open']
            prev_hour_close = self.historical_data[-1]['close']
            prev_hour_direction = 1 if prev_hour_close > prev_hour_open else 0  # 1 for UP, 0 for DOWN
            prev_hour_trend_str = "UP" if prev_hour_direction == 1 else "DOWN"
            logging.info(f"DEBUG: Previous hour trend was {prev_hour_trend_str} (Open: {prev_hour_open}, Close: {prev_hour_close}).")

            # 2. Determine current hour's 5-minute trend
            current_5min_direction = 1 if current_price > self.start_of_hour_price else 0 # 1 for UP, 0 for DOWN
            current_5min_trend_str = "UP" if current_5min_direction == 1 else "DOWN"
            logging.info(f"DEBUG: Current 5-min trend is {current_5min_trend_str} (Start: {self.start_of_hour_price}, Now: {current_price}).")

            # 3. Compare trends and make a prediction
            if prev_hour_direction == current_5min_direction:
                # Trends match, make a prediction
                prediction = prev_hour_direction
                confidence = 0.85
                explanation = f"Previous hour was {prev_hour_trend_str} and first 5 mins are {current_5min_trend_str}. Predicting {prev_hour_trend_str}."
                logging.info(f"DEBUG: Trends match. Predicting {prev_hour_trend_str}.")
            else:
                # Trends do not match, do not make a prediction
                prediction = None
                confidence = 0.0
                explanation = f"No prediction: Previous hour was {prev_hour_trend_str} but first 5 mins are {current_5min_trend_str}."
                logging.info(f"DEBUG: Trends do not match. No prediction.")

            self.last_prediction = prediction  # Store prediction to avoid multiple predictions in the same hour
            return prediction, confidence, explanation

        # If not enough time has passed since the hour started
        explanation = "Waiting for 5 minutes after the hour starts to make a prediction."
        return None, 0.0, explanation
