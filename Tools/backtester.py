import pandas as pd
import os
import importlib.util
from config.config import BACKTEST_FILE, BACKTEST_TRIGGER_MINUTE, REQUIRED_MINUTES, MODEL_FILE
from Models.predictor import PricePredictor

class Backtester:
    def __init__(self):
        # Dynamically load the model class specified in config
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Models", MODEL_FILE)
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
            
        self.predictor = predictor_class()
        print(f"Using predictor: {predictor_class.__name__} from {MODEL_FILE}")
        self.results = []

    def run_backtest(self):
        """Run complete backtesting process with the simplified predictor."""
        print("Starting backtest...")

        try:
            # Create data directory if it doesn't exist
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data")
            os.makedirs(data_dir, exist_ok=True)
            
            # Get the project root directory
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path = os.path.join(root_dir, BACKTEST_FILE)
            
            df = pd.read_csv(file_path,
                           names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
            df = df.sort_values('timestamp').reset_index(drop=True)

            if df.empty:
                print("ERROR: No data left after loading. Please check the CSV file.")
                return

            print(f"Loaded {len(df)} rows from {BACKTEST_FILE}.")

            df['hour'] = df['timestamp'].dt.floor('h')
            hourly_groups = df.groupby('hour')
            unique_hours = df['hour'].unique()

            for current_hour in unique_hours:
                group = hourly_groups.get_group(current_hour)
                print(f"\nProcessing hour: {current_hour.strftime('%Y-%m-%d %H:%M')}")
                print(f" - Total rows in hour: {len(group)}")

                if len(group) < REQUIRED_MINUTES:
                    print(f"Skipping hour {current_hour}, not enough minutes in group.")
                    continue

                if BACKTEST_TRIGGER_MINUTE >= len(group):
                    print(f"Skipping hour {current_hour}, not enough data for trigger minute.")
                    continue

                # --- Start of changes ---
                # Reset predictor state for the new hour. This is crucial for stateful models.
                self.predictor.reset_state()

                # 1. Feed the first candle of the hour to establish the starting price
                start_of_hour_candle = group.iloc[0]
                live_update_start = {
                    'timestamp': start_of_hour_candle['timestamp'],
                    'price': start_of_hour_candle['open']
                }
                # This call will set the start_of_hour_price in the predictor
                self.predictor.process_live_update(live_update_start)
                print(f" - Start of hour price recorded: {start_of_hour_candle['open']:.2f} at {start_of_hour_candle['timestamp']}")

                # 2. Feed the trigger candle to get the prediction
                trigger_candle = group.iloc[BACKTEST_TRIGGER_MINUTE]
                print(f" - Trigger candle at index {BACKTEST_TRIGGER_MINUTE}: {trigger_candle.to_dict()}")

                live_update_trigger = {
                    'timestamp': trigger_candle['timestamp'],
                    'price': trigger_candle['open']
                }

                prediction, confidence, explanation = self.predictor.process_live_update(live_update_trigger)
                # --- End of changes ---
                print(f" - Prediction: {prediction}, Confidence: {confidence}, Explanation: {explanation}")

                if prediction is not None:
                    hourly_open = group.iloc[0]['open']
                    hourly_close = group.iloc[-1]['close']
                    actual_outcome = 1 if hourly_close > hourly_open else 0
                    correct = "✅" if prediction == actual_outcome else "❌"
                    print(f" - Hourly Open: {hourly_open}, Hourly Close: {hourly_close}, Actual: {actual_outcome}")
                    print(f" - Prediction was: {'Up' if prediction == 1 else 'Down'} => {correct}")

                    self.results.append({
                        'Hour': current_hour.strftime('%Y-%m-%d %H:%M'),
                        'Hourly Open': f"{hourly_open:.2f}",
                        'Hourly Close': f"{hourly_close:.2f}",
                        'Predicted': "Up" if prediction == 1 else "Down",
                        'Actual': "Up" if actual_outcome == 1 else "Down",
                        'Confidence': f"{confidence:.2f}",
                        'Correct': correct,
                        'Explanation': explanation
                    })

            self.generate_report()

        except FileNotFoundError:
            print(f"Error: {BACKTEST_FILE} not found.")
        except Exception as e:
            print(f"Error during backtest: {str(e)}")

    def generate_report(self):
        """Generate HTML report from backtest results"""
        if not self.results:
            print("\nNo results to report")
            return

        total_predictions = len(self.results)
        correct_predictions = sum(1 for r in self.results if r['Correct'] == "✅")
        incorrect_predictions = total_predictions - correct_predictions
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

        # Calculate longest win and loss streaks
        longest_win_streak = 0
        longest_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0
        streaks = []

        for res in self.results:
            if res['Correct'] == "✅":
                current_win_streak += 1
                longest_win_streak = max(longest_win_streak, current_win_streak)
                if current_loss_streak > 0:
                    streaks.append(('Loss', current_loss_streak))
                current_loss_streak = 0
            else:
                current_loss_streak += 1
                longest_loss_streak = max(longest_loss_streak, current_loss_streak)
                if current_win_streak > 0:
                    streaks.append(('Win', current_win_streak))
                current_win_streak = 0
        # Append the last streak
        if current_win_streak > 0:
            streaks.append(('Win', current_win_streak))
        if current_loss_streak > 0:
            streaks.append(('Loss', current_loss_streak))

        # Count overall prediction distribution (Up vs Down)
        up_predictions = sum(1 for r in self.results if r['Predicted'] == "Up")
        down_predictions = total_predictions - up_predictions

        html_content = f"""
        <html>
        <head>
            <title>Backtest Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .correct {{ color: green; }}
                .incorrect {{ color: red; }}
                .dashboard {{ display: flex; gap: 40px; margin-bottom: 40px; }}
                .chart-container {{ width: 350px; }}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <h1>Backtest Results</h1>
            <h2>Summary</h2>
            <div class="dashboard">
                <div class="chart-container">
                    <canvas id="resultsPieChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="streakChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="predictionPieChart"></canvas>
                </div>
                <div>
                    <p><b>Total Predictions:</b> {total_predictions}</p>
                    <p><b>Correct Predictions:</b> {correct_predictions}</p>
                    <p><b>Incorrect Predictions:</b> {incorrect_predictions}</p>
                    <p><b>Accuracy:</b> {accuracy:.2f}%</p>
                    <p><b>Longest Win Streak:</b> {longest_win_streak}</p>
                    <p><b>Longest Loss Streak:</b> {longest_loss_streak}</p>
                </div>
            </div>
            <h2>Details</h2>
            <table>
                <tr>
                    <th>Hour</th>
                    <th>Hourly Open</th>
                    <th>Hourly Close</th>
                    <th>Predicted</th>
                    <th>Actual</th>
                    <th>Confidence</th>
                    <th>Correct</th>
                    <th>Explanation</th>
                </tr>
        """

        for res in self.results:
            correct_class = "correct" if res['Correct'] == "✅" else "incorrect"
            html_content += f"""
                <tr>
                    <td>{res['Hour']}</td>
                    <td>{res['Hourly Open']}</td>
                    <td>{res['Hourly Close']}</td>
                    <td>{res['Predicted']}</td>
                    <td>{res['Actual']}</td>
                    <td>{res['Confidence']}</td>
                    <td class="{correct_class}">{res['Correct']}</td>
                    <td>{res['Explanation']}</td>
                </tr>
            """

        html_content += """
            </table>
            <script>
                // Pie Chart: Correct vs Incorrect
                const resultsCtx = document.getElementById('resultsPieChart').getContext('2d');
                new Chart(resultsCtx, {
                    type: 'pie',
                    data: {
                        labels: ['Correct', 'Incorrect'],
                        datasets: [{
                            data: [""" + f"{correct_predictions}, {incorrect_predictions}" + """],
                            backgroundColor: ['#4caf50', '#f44336'],
                        }]
                    },
                    options: {
                        plugins: {
                            legend: { display: true, position: 'bottom' },
                            title: { display: true, text: 'Prediction Accuracy' }
                        }
                    }
                });

                // Bar Chart: Longest Win/Loss Streaks
                const streakCtx = document.getElementById('streakChart').getContext('2d');
                new Chart(streakCtx, {
                    type: 'bar',
                    data: {
                        labels: ['Longest Win Streak', 'Longest Loss Streak'],
                        datasets: [{
                            label: 'Streak Length',
                            data: [""" + f"{longest_win_streak}, {longest_loss_streak}" + """],
                            backgroundColor: ['#2196f3', '#ff9800'],
                        }]
                    },
                    options: {
                        plugins: {
                            legend: { display: false },
                            title: { display: true, text: 'Longest Streaks' }
                        },
                        scales: {
                            y: { beginAtZero: true, precision: 0 }
                        }
                    }
                });

                // New Pie Chart: Distribution of Predictions (Up vs Down)
                const predictionCtx = document.getElementById('predictionPieChart').getContext('2d');
                new Chart(predictionCtx, {
                    type: 'pie',
                    data: {
                        labels: ['Up', 'Down'],
                        datasets: [{
                            data: [""" + f"{up_predictions}, {down_predictions}" + """],
                            backgroundColor: ['#8bc34a', '#03a9f4'],
                        }]
                    },
                    options: {
                        plugins: {
                            legend: { display: true, position: 'bottom' },
                            title: { display: true, text: 'Prediction Direction' }
                        }
                    }
                });
            </script>
        </body>
        </html>
        """

        # Create Output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Output")
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "backtest_results.html")
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        print(f"\nBacktest complete. Results saved to {output_file}")
        print(f"Accuracy: {accuracy:.2f}% ({correct_predictions} correct out of {total_predictions})")
