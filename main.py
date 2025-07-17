import os
import importlib.util
from Data.data_fetcher import BinanceDataFetcher
from Models.predictor import PricePredictor, CONFIDENCE_THRESHOLD, MAX_WAIT_MINUTES
from Tools.backtester import Backtester
from config.config import MODEL_FILE


def main():
    print("🚀 BTC Price Predictor v2")
    print("=" * 40)

    choice = input("Run backtest or live prediction? (backtest/live): ").lower()

    if choice == 'backtest':
        backtester = Backtester()
        backtester.run_backtest()
        return

    # Initialize components
    data_fetcher = BinanceDataFetcher()

    # Dynamically load the model class specified in config
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models", MODEL_FILE)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: Model file '{MODEL_FILE}' not found in the Models directory. "
                               f"Please check your MODEL_FILE setting in config.py.")
        
    # Import the module and get the predictor class
    spec = importlib.util.spec_from_file_location("dynamic_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get the first class defined in the module that is a subclass of PricePredictor
    predictor_class = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, PricePredictor) and attr != PricePredictor:
            predictor_class = attr
            break
    
    if not predictor_class:
        raise ValueError(f"Error: No PricePredictor subclass found in '{MODEL_FILE}'. "
                        f"Your model file must contain at least one class that inherits from PricePredictor. "
                        f"Example: class YourPredictor(PricePredictor): ...")
        
    predictor = predictor_class()
    print(f"Using predictor: {predictor_class.__name__} from {MODEL_FILE}")

    # Fetch historical data
    print("\n📊 Step 1: Fetching historical data...")
    historical_df = data_fetcher.fetch_past_data()

    if historical_df.empty:
        print("No historical data fetched. Exiting.")
        return

    predictor.load_historical_data(historical_df)

    print("\n🔮 Step 2: Starting live updates and predictions...")

    # Track predictions during live session
    predictions_made = 0
    confident_predictions = 0
    last_prediction_time = None

    def live_update_callback(update_data):
        nonlocal predictions_made, confident_predictions, last_prediction_time

        current_time = update_data['timestamp']
        current_price = update_data['open'] # Use open price for consistency with backtester

        # Calculate minutes since last prediction
        if last_prediction_time:
            minutes_since_prediction = (current_time - last_prediction_time).seconds / 60
        else:
            minutes_since_prediction = float('inf')

        # Process update and get prediction
        # Pass the entire OHLCV dictionary to the predictor
        prediction, confidence, explanation = predictor.process_live_update(update_data)

        if prediction is not None:
            predictions_made += 1

            if confidence >= CONFIDENCE_THRESHOLD:
                confident_predictions += 1
                direction = "up (⬆)" if prediction == 1 else "down (⬇)"
                print(f"\n🔮 Prediction: Next candle close will be {direction} (confidence: {confidence:.2f})")
                print(f"Reason: {explanation}")
                print(f"Current price: {current_price:.2f}")
                last_prediction_time = current_time
            else:
                # Only print waiting message if we haven't made a prediction recently
                if minutes_since_prediction > MAX_WAIT_MINUTES or last_prediction_time is None:
                    print("\nWaiting for clearer signal...")
                    print(f"Current price: {current_price:.2f}")
                    print(f"Confidence too low ({confidence:.2f} < {CONFIDENCE_THRESHOLD:.2f})")

    try:
        data_fetcher.start_live_updates(callback=live_update_callback)
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
        print(f"\nSession stats: {predictions_made} predictions made, {confident_predictions} with sufficient confidence")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
