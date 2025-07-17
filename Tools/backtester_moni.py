import pandas as pd
import os
import importlib.util
from config.config import BACKTEST_FILE, BACKTEST_TRIGGER_MINUTE, REQUIRED_MINUTES, MODEL_FILE
from Models.predictor import PricePredictor

# need to FIX FIX FIX
class Backtester:
    # Money management variables (Martingale strategy)
    INITIAL_CAPITAL = 1000.0  # Starting capital amount ($)
    TRADE_CAPITAL_PERCENT = 0.01220703125  # Percentage of capital to use per trade
    MARTINGALE_MULTIPLIER = 2.0  # Multiplier after a loss (classic martingale = 2)
    RESET_ON_WIN = True  # Reset bet amount after a win

    def __init__(self, initial_capital=None, trade_capital_percent=None,
                 martingale_multiplier=None, reset_on_win=None):
        # Set money management parameters
        self.initial_capital = initial_capital if initial_capital is not None else self.INITIAL_CAPITAL
        self.trade_capital_percent = trade_capital_percent if trade_capital_percent is not None else self.TRADE_CAPITAL_PERCENT
        self.martingale_multiplier = martingale_multiplier if martingale_multiplier is not None else self.MARTINGALE_MULTIPLIER
        self.reset_on_win = reset_on_win if reset_on_win is not None else self.RESET_ON_WIN

        # Trading metrics
        self.current_capital = self.initial_capital
        self.current_bet_amount = (self.initial_capital * self.trade_capital_percent / 100)
        self.max_capital = self.initial_capital
        self.min_capital = self.initial_capital
        self.capital_history = []
        self.bet_history = []

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
        print(f"Initial capital: ${self.initial_capital:.2f}")
        print(f"Capital per trade: {self.trade_capital_percent:.2f}%")
        print(f"Martingale multiplier: {self.martingale_multiplier:.2f}x")
        print(f"Reset on win: {self.reset_on_win}")
        print("Proceeding with backtest...")

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

            # Reset trading metrics
            self.current_capital = self.initial_capital
            self.current_bet_amount = (self.initial_capital * self.trade_capital_percent / 100)
            self.max_capital = self.initial_capital
            self.min_capital = self.initial_capital
            self.capital_history = [self.initial_capital]
            self.bet_history = []

            for current_hour in unique_hours:
                # Check if capital has reached zero, stop backtest if it has
                if self.current_capital <= 0:
                    print("\n*** STOPPING BACKTEST: Capital has reached zero! ***")
                    break

                group = hourly_groups.get_group(current_hour)
                print(f"\nProcessing hour: {current_hour.strftime('%Y-%m-%d %H:%M')}")
                print(f" - Total rows in hour: {len(group)}")

                if len(group) < REQUIRED_MINUTES:
                    print(f"Skipping hour {current_hour}, not enough minutes in group.")
                    continue

                if BACKTEST_TRIGGER_MINUTE >= len(group):
                    print(f"Skipping hour {current_hour}, not enough data for trigger minute.")
                    continue

                # Reset predictor state for the new hour. This is crucial for stateful models.
                self.predictor.reset_state()

                prediction, confidence, explanation = None, 0.0, "No prediction made."

                # Simulate minute-by-minute updates for the first part of the hour
                # This allows stateful predictors (like volume accumulators) to work correctly.
                for i in range(BACKTEST_TRIGGER_MINUTE + 1):
                    candle = group.iloc[i]
                    candle_data = candle.to_dict()

                    # The final prediction is taken from the trigger minute
                    prediction, confidence, explanation = self.predictor.process_live_update(candle_data)

                    if i == 0:
                        print(f" - Start of hour price recorded: {candle['open']:.2f} at {candle['timestamp']}")
                    if i == BACKTEST_TRIGGER_MINUTE:
                        print(f" - Trigger candle at index {i}: {candle_data}")

                print(f" - Prediction: {prediction}, Confidence: {confidence}, Explanation: {explanation}")

                if prediction is not None:
                    hourly_open = group.iloc[0]['open']
                    hourly_close = group.iloc[-1]['close']
                    actual_outcome = 1 if hourly_close > hourly_open else 0
                    correct = "✅" if prediction == actual_outcome else "❌"

                    # Ensure the bet doesn't exceed available capital
                    bet_amount = min(self.current_bet_amount, self.current_capital)

                    # Calculate PnL for this trade
                    if prediction == actual_outcome:  # Correct prediction
                        profit = bet_amount  # 100% profit (bet amount doubles)
                        # Apply Martingale strategy: reset bet after a win if configured
                        if self.reset_on_win:
                            self.current_bet_amount = (self.current_capital * self.trade_capital_percent / 100)
                        pnl_str = f"+${profit:.2f}"
                    else:  # Incorrect prediction
                        profit = -bet_amount  # Lose entire bet
                        # Apply Martingale strategy: double bet after a loss
                        self.current_bet_amount *= self.martingale_multiplier
                        pnl_str = f"-${abs(profit):.2f}"

                    # Update capital
                    self.current_capital += profit
                    self.max_capital = max(self.max_capital, self.current_capital)
                    self.min_capital = min(self.min_capital, self.current_capital)
                    self.capital_history.append(self.current_capital)
                    self.bet_history.append(bet_amount)

                    print(f" - Hourly Open: {hourly_open}, Hourly Close: {hourly_close}, Actual: {actual_outcome}")
                    print(f" - Prediction was: {'Up' if prediction == 1 else 'Down'} => {correct}")
                    print(f" - Trade: Bet ${bet_amount:.2f}, PnL: {pnl_str}, New Capital: ${self.current_capital:.2f}")

                    self.results.append({
                        'Hour': current_hour.strftime('%Y-%m-%d %H:%M'),
                        'Hourly Open': f"{hourly_open:.2f}",
                        'Hourly Close': f"{hourly_close:.2f}",
                        'Predicted': "Up" if prediction == 1 else "Down",
                        'Actual': "Up" if actual_outcome == 1 else "Down",
                        'Confidence': f"{confidence:.2f}",
                        'Correct': correct,
                        'Bet Amount': f"${bet_amount:.2f}",
                        'PnL': pnl_str,
                        'Capital': f"${self.current_capital:.2f}",
                        'Explanation': explanation
                    })

                    # Check if capital has reached zero after the trade
                    if self.current_capital <= 0:
                        print("\n*** STOPPING BACKTEST: Capital has reached zero! ***")
                        break

            # Generate report regardless of capital status
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

        # Calculate financial metrics
        final_capital = self.current_capital
        profit_loss = final_capital - self.initial_capital
        roi_percent = (profit_loss / self.initial_capital) * 100
        profit_factor = 0
        if correct_predictions > 0 and incorrect_predictions > 0:
            avg_win = sum(
                float(r['PnL'].replace('+$', '')) for r in self.results if r['Correct'] == "✅") / correct_predictions
            avg_loss = sum(
                float(r['PnL'].replace('-$', '')) for r in self.results if r['Correct'] == "❌") / incorrect_predictions
            if avg_loss > 0:  # Avoid division by zero
                profit_factor = avg_win / avg_loss

        # Maximum drawdown calculation
        max_drawdown_pct = 0
        peak = self.initial_capital
        for capital in self.capital_history:
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak * 100
            max_drawdown_pct = max(max_drawdown_pct, drawdown)

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
                .positive {{ color: green; font-weight: bold; }}
                .negative {{ color: red; font-weight: bold; }}
                .dashboard {{ display: flex; flex-wrap: wrap; gap: 40px; margin-bottom: 40px; }}
                .chart-container {{ width: 350px; }}
                .metrics-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .metric-box {{ 
                    border: 1px solid #ddd; 
                    border-radius: 5px; 
                    padding: 15px; 
                    width: 200px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-title {{ font-weight: bold; margin-bottom: 5px; color: #555; }}
                .metric-value {{ font-size: 24px; margin-bottom: 10px; }}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <h1>Backtest Results</h1>

            <h2>Financial Summary</h2>
            <div class="metrics-container">
                <div class="metric-box">
                    <div class="metric-title">Initial Capital</div>
                    <div class="metric-value">${self.initial_capital:.2f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">Final Capital</div>
                    <div class="metric-value">${final_capital:.2f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">Profit/Loss</div>
                    <div class="metric-value {'positive' if profit_loss >= 0 else 'negative'}">${profit_loss:.2f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">ROI</div>
                    <div class="metric-value {'positive' if roi_percent >= 0 else 'negative'}">{roi_percent:.2f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">Max Capital</div>
                    <div class="metric-value">${self.max_capital:.2f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">Min Capital</div>
                    <div class="metric-value">${self.min_capital:.2f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">Max Drawdown</div>
                    <div class="metric-value negative">{max_drawdown_pct:.2f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">Profit Factor</div>
                    <div class="metric-value">{profit_factor:.2f}</div>
                </div>
            </div>

            <h2>Prediction Summary</h2>
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
                <div class="chart-container">
                    <canvas id="capitalChart"></canvas>
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

            <h2>Trade Details</h2>
            <table>
                <tr>
                    <th>Hour</th>
                    <th>Hourly Open</th>
                    <th>Hourly Close</th>
                    <th>Predicted</th>
                    <th>Actual</th>
                    <th>Bet Amount</th>
                    <th>PnL</th>
                    <th>Capital</th>
                    <th>Confidence</th>
                    <th>Correct</th>
                    <th>Explanation</th>
                </tr>
        """

        for res in self.results:
            correct_class = "correct" if res['Correct'] == "✅" else "incorrect"
            pnl_class = "positive" if res['PnL'].startswith('+') else "negative"
            html_content += f"""
                <tr>
                    <td>{res['Hour']}</td>
                    <td>{res['Hourly Open']}</td>
                    <td>{res['Hourly Close']}</td>
                    <td>{res['Predicted']}</td>
                    <td>{res['Actual']}</td>
                    <td>{res['Bet Amount']}</td>
                    <td class="{pnl_class}">{res['PnL']}</td>
                    <td>{res['Capital']}</td>
                    <td>{res['Confidence']}</td>
                    <td class="{correct_class}">{res['Correct']}</td>
                    <td>{res['Explanation']}</td>
                </tr>
            """

        # JavaScript for charts
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


                // Pie Chart: Distribution of Predictions (Up vs Down)
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

                // Line Chart: Capital History
                const capitalCtx = document.getElementById('capitalChart').getContext('2d');
                new Chart(capitalCtx, {
                    type: 'line',
                    data: {
                        labels: Array.from({length: """ + f"{len(self.capital_history)}" + """}, (_, i) => i),
                        datasets: [{
                            label: 'Capital',
                            data: """ + f"{self.capital_history}" + """,
                            borderColor: '#4caf50',
                            tension: 0.1,
                            fill: false
                        }]
                    },
                    options: {
                        plugins: {
                            title: { display: true, text: 'Capital History' }
                        },
                        scales: {
                            y: { 
                                beginAtZero: false,
                                title: {
                                    display: true,
                                    text: 'Capital ($)'
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Trade #'
                                }
                            }
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
        print(f"Final capital: ${self.current_capital:.2f} (ROI: {roi_percent:.2f}%)")
