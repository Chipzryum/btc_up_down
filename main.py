from Data.data_fetcher import BinanceDataFetcher
from Models.predictor import PricePredictor, CONFIDENCE_THRESHOLD, MAX_WAIT_MINUTES
from Tools.backtester import Backtester


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
    predictor = PricePredictor()

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
        current_minute = current_time.minute

        # Calculate minutes since last prediction
        if last_prediction_time:
            minutes_since_prediction = (current_time - last_prediction_time).seconds / 60
        else:
            minutes_since_prediction = float('inf')

        # Process update and get prediction
        prediction, confidence, explanation = predictor.process_live_update(update_data)

        if prediction is not None:
            predictions_made += 1

            if confidence >= CONFIDENCE_THRESHOLD:
                confident_predictions += 1
                direction = "up (⬆)" if prediction == 1 else "down (⬇)"
                print(f"\n🔮 Prediction: Next candle close will be {direction} (confidence: {confidence:.2f})")
                print(f"Reason: {explanation}")
                print(f"Current price: {update_data['price']:.2f}")
                last_prediction_time = current_time
            else:
                # Only print waiting message if we haven't made a prediction recently
                if minutes_since_prediction > MAX_WAIT_MINUTES or last_prediction_time is None:
                    print("\nWaiting for clearer signal...")
                    print(f"Current price: {update_data['price']:.2f}")
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
