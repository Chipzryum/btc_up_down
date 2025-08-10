import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Models.predictor import PricePredictor

class AlwaysUpPredictor(PricePredictor):
    def __init__(self, opposite=False):
        super().__init__()
        # If True, flip the usual UP prediction to DOWN
        self.opposite = opposite
        self.historical_data = []

    def reset_state(self):
        self.historical_data = []

    def load_historical_data(self, historical_data):
        self.historical_data = historical_data

    def make_prediction(self):
        base_prediction = 1          # 1 == UP
        # invert only if opposite is set
        prediction = 1 - base_prediction if self.opposite else base_prediction
        confidence = 0.9

        if self.opposite:
            explanation = (
                "Always Up Predictor (Inverted): Consistently predicting DOWN "
                "regardless of market conditions"
            )
        else:
            explanation = (
                "Always Up Predictor: Consistently predicting UP "
                "regardless of market conditions"
            )

        return prediction, confidence, explanation
