import sys
import os
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Models.predictor import PricePredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EnhancedPredictor(PricePredictor):
    """
    V5 - The Exhaustion & Continuation Model
    This model correctly interprets the 'Conviction Score' from V4.
    - A HIGH score on a candle implies EXHAUSTION and a likely REVERSAL.
    - A LOW score implies a weak pause and a likely CONTINUATION.
    """

    def __init__(self):
        super().__init__()
        self.historical_data: List[Dict] = []
        self.PREV_DATA_COUNT = 25

        # --- Indicator Parameters ---
        self.volume_ma_period = 20
        self.ema_trend_period = 9
        self.bollinger_period = 20
        self.conviction_threshold = 2  # Min score to be considered a high-conviction (exhaustion) move

    def reset_state(self):
        self.historical_data = []

    def load_historical_data(self, historical_data: List[Dict]):
        self.historical_data = historical_data[-self.PREV_DATA_COUNT:]

    def _calculate_technical_indicators(self, data: List[Dict]) -> Optional[pd.DataFrame]:
        if len(data) < self.PREV_DATA_COUNT:
            return None

        df = pd.DataFrame(data).copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['Volume_MA'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        df['EMA_Trend'] = df['close'].ewm(span=self.ema_trend_period, adjust=False).mean()
        df['BB_Mid'] = df['close'].rolling(window=self.bollinger_period).mean()
        df['Body_Ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)

        return df.dropna().reset_index(drop=True)

    def make_prediction(self) -> Tuple[Optional[int], float, str]:
        if not self.historical_data or len(self.historical_data) < self.PREV_DATA_COUNT:
            return None, 0.0, "Not enough historical data."

        df_indicators = self._calculate_technical_indicators(self.historical_data)

        if df_indicators is None or df_indicators.empty:
            return None, 0.0, "Not enough valid data after calculating indicators."

        latest_data = df_indicators.iloc[-1]
        return self._predict_based_on_market_psychology(latest_data)

    def _predict_based_on_market_psychology(self, latest_data: pd.Series) -> Tuple[int, float, str]:
        close = latest_data['close']
        open_price = latest_data['open']

        if close > open_price:
            base_signal = 1  # UP
            direction_str = "UP"
        else:
            base_signal = 0  # DOWN
            direction_str = "DOWN"

        conviction_score = 0
        reasons = []

        if latest_data['volume'] > latest_data['Volume_MA']:
            conviction_score += 1
            reasons.append("High Volume")
        if latest_data['Body_Ratio'] > 0.6:
            conviction_score += 1
            reasons.append("Strong Body")
        is_above_trend = close > latest_data['EMA_Trend']
        if (base_signal == 1 and is_above_trend) or (base_signal == 0 and not is_above_trend):
            conviction_score += 1
            reasons.append("Trend Agrees")
        is_above_mid = close > latest_data['BB_Mid']
        if (base_signal == 1 and is_above_mid) or (base_signal == 0 and not is_above_mid):
            conviction_score += 1
            reasons.append("Momentum Agrees")

        # --- CORRECTED PREDICTION LOGIC ---
        # This logic now matches your profitable "Reversed V4" results.
        if conviction_score >= self.conviction_threshold:
            # HIGH score = EXHAUSTION. We bet on a REVERSAL.
            prediction = 1 - base_signal
            confidence = 0.55 + (conviction_score * 0.05)
            explanation = f"Exhaustion Reversal on strong {direction_str} (Score: {conviction_score}/4)"
        else:
            # LOW score = WEAK PAUSE. We bet on a CONTINUATION.
            prediction = base_signal
            confidence = 0.52
            explanation = f"Continuation on weak {direction_str} (Score: {conviction_score}/4)"

        return prediction, round(confidence, 4), explanation


# Example usage for direct testing
if __name__ == "__main__":
    predictor = EnhancedPredictor()
    print("EnhancedPredictor (V5 - Exhaustion & Continuation Model) is defined and ready for backtesting.")