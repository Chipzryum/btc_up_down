import pandas as pd
import os
import importlib.util
from config.config import BACKTEST_FILE, MODEL_FILE
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
                # Also get historical data requirements from the model file
                self.prev_data_count = getattr(module, 'PREV_DATA_COUNT', 0)
                break

        if not predictor_class:
            raise ValueError(f"Error: No PricePredictor subclass found in '{MODEL_FILE}'. "
                             f"Your model file must contain at least one class that inherits from PricePredictor. "
                             f"Example: class YourPredictor(PricePredictor): ...")

        self.predictor = predictor_class()
        self.model_name = predictor_class.__name__
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

            # Read CSV with all columns, then select only the required ones
            df = pd.read_csv(file_path)
            # Select only the columns needed for backtesting
            required_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
            df = df[required_cols]
            df['open_time'] = pd.to_datetime(df['open_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            df.dropna(subset=['open_time'], inplace=True)
            df = df.sort_values('open_time').reset_index(drop=True)

            if df.empty:
                print("ERROR: No data left after loading. Please check the CSV file.")
                return

            print(f"Loaded {len(df)} rows from {BACKTEST_FILE}.")

            df['hour'] = df['open_time'].dt.floor('h')
            hourly_groups = df.groupby('hour')
            unique_hours = df['hour'].unique()

            historical_candles_buffer = []
            self.predictor.reset_state()  # Reset state once before the loop

            for i in range(len(unique_hours)):
                current_hour = unique_hours[i]
                group = hourly_groups.get_group(current_hour)
                print(f"\nProcessing hour: {current_hour.strftime('%Y-%m-%d %H:%M')}")

                # The prediction for the current hour is based on all data *before* it.
                # Load all available historical candles into the model.
                self.predictor.load_historical_data(historical_candles_buffer)

                # Make a prediction for the current hour.
                prediction, confidence, explanation = self.predictor.make_prediction()

                print(f" - Prediction for this hour: {prediction}, Confidence: {confidence}, Explanation: {explanation}")

                if prediction is not None:
                    # The actual outcome is determined by the current hour's candle
                    hourly_open = group.iloc[0]['open']
                    hourly_close = group.iloc[0]['close']
                    actual_outcome = 1 if hourly_close > hourly_open else 0
                    correct = "✅" if prediction == actual_outcome else "❌"
                    print(f" - Actual:   Open: {hourly_open}, Close: {hourly_close}, Outcome: {'Up' if actual_outcome == 1 else 'Down'}")
                    print(f" - Predicted: {'Up' if prediction == 1 else 'Down'} => {correct}")

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

                # Add the processed hour's candle to the historical buffer for the next iteration
                historical_candles_buffer.append(group.iloc[0].to_dict())

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

        # --- Initial Data Processing ---
        # Create a DataFrame for easier and more efficient calculations
        results_df = pd.DataFrame(self.results)
        results_df['Hour'] = pd.to_datetime(results_df['Hour'])
        results_df.set_index('Hour', inplace=True)
        results_df['Correct_Int'] = results_df['Correct'].apply(lambda x: 1 if x == "✅" else 0)
        results_df['Predicted_Int'] = results_df['Predicted'].apply(lambda x: 1 if x == "Up" else 0)
        results_df['Actual_Int'] = results_df['Actual'].apply(lambda x: 1 if x == "Up" else 0)

        # --- Basic Stats ---
        total_predictions = len(results_df)
        correct_predictions = int(results_df['Correct_Int'].sum())
        incorrect_predictions = total_predictions - correct_predictions
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

        # --- Confusion Matrix and Advanced Metrics (New) ---
        TP = len(results_df[(results_df['Predicted_Int'] == 1) & (results_df['Actual_Int'] == 1)])
        TN = len(results_df[(results_df['Predicted_Int'] == 0) & (results_df['Actual_Int'] == 0)])
        FP = len(results_df[(results_df['Predicted_Int'] == 1) & (results_df['Actual_Int'] == 0)])
        FN = len(results_df[(results_df['Predicted_Int'] == 0) & (results_df['Actual_Int'] == 1)])

        # Precision, Recall, F1 for "Up" predictions (Positive Class)
        precision_up = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall_up = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_up = 2 * (precision_up * recall_up) / (precision_up + recall_up) if (precision_up + recall_up) > 0 else 0

        # Specificity and metrics for "Down" predictions (Negative Class)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0  # How well we identify actual "Down"s
        precision_down = TN / (TN + FN) if (TN + FN) > 0 else 0
        f1_down = 2 * (precision_down * specificity) / (precision_down + specificity) if (
                                                                                                     precision_down + specificity) > 0 else 0

        # Matthews Correlation Coefficient (MCC): A balanced measure for binary classification
        mcc_numerator = (TP * TN) - (FP * FN)
        mcc_denominator = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
        mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0

        # No-Information Rate (Baseline Accuracy to beat)
        total_actual_up = TP + FN
        total_actual_down = TN + FP
        most_frequent_class_count = max(total_actual_up, total_actual_down)
        no_info_rate = (most_frequent_class_count / total_predictions) * 100 if total_predictions > 0 else 0
        accuracy_vs_baseline = accuracy - no_info_rate

        # --- Streak Analysis ---
        longest_win_streak, longest_loss_streak, avg_win_streak, avg_loss_streak = 0, 0, 0, 0
        current_win_streak, current_loss_streak = 0, 0
        streaks, loss_streaks, win_streaks = [], [], []

        for correct in results_df['Correct_Int']:
            if correct == 1:
                current_win_streak += 1
                if current_loss_streak > 0:
                    streaks.append(('Loss', current_loss_streak))
                    loss_streaks.append(current_loss_streak)
                current_loss_streak = 0
            else:
                current_loss_streak += 1
                if current_win_streak > 0:
                    streaks.append(('Win', current_win_streak))
                    win_streaks.append(current_win_streak)
                current_win_streak = 0
        if current_win_streak > 0: win_streaks.append(current_win_streak)
        if current_loss_streak > 0: loss_streaks.append(current_loss_streak)

        longest_win_streak = max(win_streaks) if win_streaks else 0
        longest_loss_streak = max(loss_streaks) if loss_streaks else 0
        avg_win_streak = sum(win_streaks) / len(win_streaks) if win_streaks else 0
        avg_loss_streak = sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0
        loss_streak_frequency = total_predictions / len(loss_streaks) if loss_streaks else float('inf')
        win_loss_ratio = correct_predictions / incorrect_predictions if incorrect_predictions > 0 else float('inf')
        consistency_ratio = avg_win_streak / longest_loss_streak if longest_loss_streak > 0 else float('inf')

        # Top 3 longest loss streaks
        top_3_loss_streaks = sorted(loss_streaks, reverse=True)[:3]
        while len(top_3_loss_streaks) < 3: top_3_loss_streaks.append(0)

        # Loss Streak Distribution Chart
        loss_streak_dist = {i: loss_streaks.count(i) for i in range(1, longest_loss_streak + 1)}
        loss_dist_labels = list(loss_streak_dist.keys())
        loss_dist_values = list(loss_streak_dist.values())

        # Theoretical probability of a loss streak
        prob_loss = (incorrect_predictions / total_predictions) if total_predictions > 0 else 0
        prob_2_losses = prob_loss ** 2
        prob_5_losses = prob_loss ** 5
        prob_10_losses = prob_loss ** 10

        # --- Chart Data Generation ---
        # Rolling Accuracy Chart Data (Updated with Resampling)
        rolling_accuracy_window = 50
        rolling_accuracy_series = results_df['Correct_Int'].rolling(window=rolling_accuracy_window,
                                                                    min_periods=1).mean() * 100

        duration_days = (results_df.index.max() - results_df.index.min()).days if not results_df.empty else 0
        if (duration_days > 730):  # > 2 years -> Weekly
            resample_rule, resample_label = 'W', "Weekly Avg"
        elif (duration_days > 180):  # > 6 months -> Daily
            resample_rule, resample_label = 'D', "Daily Avg"
        elif (duration_days > 30):  # > 1 month -> 12-Hourly
            resample_rule, resample_label = '12H', "12-Hour Avg"
        else:  # For short backtests, don't resample
            resample_rule, resample_label = None, "Per Prediction"

        if resample_rule:
            resampled_accuracy = rolling_accuracy_series.resample(resample_rule).mean().dropna()
        else:
            resampled_accuracy = rolling_accuracy_series

        rolling_accuracy_labels = resampled_accuracy.index.strftime('%Y-%m-%d %H:%M').tolist()
        rolling_accuracy_data = [f'{val:.2f}' for val in resampled_accuracy.values]
        rolling_accuracy_chart_title = f'Rolling Accuracy ({rolling_accuracy_window}-pred window, {resample_label})'

        # Prediction distribution stats
        up_predictions = int(results_df['Predicted_Int'].sum())
        down_predictions = total_predictions - up_predictions
        up_accuracy = (TP / up_predictions) * 100 if up_predictions > 0 else 0
        down_accuracy = (TN / down_predictions) * 100 if down_predictions > 0 else 0

        # --- HTML Report Generation ---
        html_content = f"""
        <html>
        <head>
            <title>Backtest Results - {self.model_name}</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px; background-color: #f8f9fa; color: #212529; }}
                h1, h2, h3, h4 {{ color: #343a40; }}
                table {{ border-collapse: collapse; width: 100%; box-shadow: 0 2px 3px rgba(0,0,0,0.1); }}
                th, td {{ border: 1px solid #dee2e6; padding: 10px; text-align: left; }}
                th {{ background-color: #e9ecef; }}
                tr:nth-child(even) {{ background-color: #f8f9fa; }}
                .correct {{ color: #28a745; font-weight: bold; }}
                .incorrect {{ color: #dc3545; font-weight: bold; }}
                .dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-bottom: 25px; }}
                .chart-container {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
                .stats-container {{ display: flex; flex-wrap: wrap; gap: 20px; width: 100%; }}
                .stats-column {{ flex: 1; min-width: 250px; }}
                .section-box {{ background-color: #fff; border: 1px solid #dee2e6; padding: 20px; margin-top: 25px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
                .metric-value {{ font-size: 1.1em; font-weight: 500; color: #007bff; }}
                .metric-label {{ font-weight: bold; }}
                .metric-explanation {{ font-size: 0.85em; color: #6c757d; margin-top: 15px; }}
                hr {{ border: none; border-top: 1px solid #dee2e6; margin: 15px 0; }}
                .loss-streak-table {{ width: auto; margin-top: 10px; }}
                .loss-streak-table th, .loss-streak-table td {{ text-align: center; }}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <h1>Backtest Results: {self.model_name}</h1>

            <div class="section-box">
                <h2>Overall Performance</h2>
                <div class="stats-container">
                    <div class="stats-column">
                        <p><span class="metric-label">Total Predictions:</span> <span class="metric-value">{total_predictions}</span></p>
                        <p><span class="metric-label">Overall Accuracy:</span> <span class="metric-value">{accuracy:.2f}%</span></p>
                        <p><span class="metric-label">Win / Loss Ratio:</span> <span class="metric-value">{win_loss_ratio:.2f} : 1</span></p>
                        <p><span class="metric-label">Correct / Incorrect:</span> {correct_predictions} / {incorrect_predictions}</p>
                    </div>
                    <div class="stats-column">
                        <p><span class="metric-label">Baseline (No-Info Rate):</span> <span class="metric-value">{no_info_rate:.2f}%</span></p>
                        <p><span class="metric-label">Edge vs Baseline:</span> <span class="metric-value" style="color: {'#28a745' if accuracy_vs_baseline > 0 else '#dc3545'};">{accuracy_vs_baseline:+.2f} pts</span></p>
                        <p><span class="metric-label">Matthews Corr. Coeff (MCC):</span> <span class="metric-value">{mcc:.3f}</span></p>
                    </div>
                </div>
                <p class="metric-explanation">
                    <b>Baseline:</b> Accuracy from always guessing the most common outcome. Your model's accuracy should be higher. <br>
                    <b>MCC:</b> A score from -1 (perfectly wrong) to +1 (perfectly right), with 0 being random. It's a robust measure of prediction quality.
                </p>
            </div>

            <div class="dashboard">
                 <div class="chart-container">
                    <canvas id="accuracyLineChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="resultsPieChart"></canvas>
                </div>
            </div>

            <div class="section-box">
                <h2>Classification Details</h2>
                <div class="stats-container">
                    <div class="stats-column">
                        <h4>"Up" Predictions ({up_predictions} total)</h4>
                        <p><span class="metric-label">Accuracy:</span> {up_accuracy:.2f}%</p>
                        <p><span class="metric-label">Precision:</span> {precision_up:.2%} <small>(Correctness of 'Up' predictions)</small></p>
                        <p><span class="metric-label">Recall:</span> {recall_up:.2%} <small>(% of actual 'Up's found)</small></p>
                        <p><span class="metric-label">F1-Score:</span> {f1_up:.3f}</p>
                    </div>
                    <div class="stats-column">
                        <h4>"Down" Predictions ({down_predictions} total)</h4>
                        <p><span class="metric-label">Accuracy:</span> {down_accuracy:.2f}%</p>
                        <p><span class="metric-label">Precision:</span> {precision_down:.2%} <small>(Correctness of 'Down' predictions)</small></p>
                        <p><span class="metric-label">Specificity:</span> {specificity:.2%} <small>(% of actual 'Down's found)</small></p>
                        <p><span class="metric-label">F1-Score:</span> {f1_down:.3f}</p>
                    </div>
                </div>
            </div>

            <div class="section-box">
                <h2>Streak Analysis</h2>
                <div class="stats-container">
                    <div class="stats-column">
                        <h4>Win Streaks</h4>
                        <p><span class="metric-label">Longest Win Streak:</span> {longest_win_streak}</p>
                        <p><span class="metric-label">Average Win Streak:</span> {avg_win_streak:.2f}</p>
                    </div>
                    <div class="stats-column">
                        <h4>Loss Streaks</h4>
                        <p><span class="metric-label">Longest Loss Streak:</span> {longest_loss_streak}</p>
                        <p><span class="metric-label">Average Loss Streak:</span> {avg_loss_streak:.2f}</p>
                        <p><span class="metric-label">Top 3 Loss Streaks:</span> {top_3_loss_streaks[0]}, {top_3_loss_streaks[1]}, {top_3_loss_streaks[2]}</p>
                    </div>
                    <div class="stats-column">
                        <h4>Consistency</h4>
                        <p><span class="metric-label">Consistency Ratio:</span> <span class="metric-value">{consistency_ratio:.2f}</span></p>
                        <p class="metric-explanation" style="margin-top: 5px;">(Avg Win Streak / Longest Loss Streak). Higher is better.</p>
                         <hr>
                        <h4>Theoretical Risk</h4>
                        <p><small>Based on a {prob_loss:.2%} loss probability:</small></p>
                        <p><small> - Chance of 5 losses in a row: {prob_5_losses:.4%}</small></p>
                        <p><small> - Chance of 10 losses in a row: {prob_10_losses:.8%}</small></p>
                    </div>
                </div>
                <div style="margin-top: 20px;">
                    <h4>Loss Streak Frequency Table</h4>
                    <table class="loss-streak-table">
                        <tr>
                            <th>Loss Streak Length</th>
                            <th>Frequency</th>
                        </tr>
                        {''.join(f'<tr><td>{length}</td><td>{count}</td></tr>' for length, count in loss_streak_dist.items())}
                    </table>
                </div>
            </div>

            <div class="section-box">
                <h2>Detailed Log</h2>
                <table>
                    <tr><th>Hour</th><th>Open</th><th>Close</th><th>Predicted</th><th>Actual</th><th>Confidence</th><th>Correct</th><th>Explanation</th></tr>
        """

        for _, res in results_df.iterrows():
            correct_class = "correct" if res['Correct'] == "✅" else "incorrect"
            html_content += f"""
                <tr>
                    <td>{res.name.strftime('%Y-%m-%d %H:%M')}</td><td>{res['Hourly Open']}</td><td>{res['Hourly Close']}</td><td>{res['Predicted']}</td>
                    <td>{res['Actual']}</td><td>{res['Confidence']}</td><td class="{correct_class}">{res['Correct']}</td><td>{res['Explanation']}</td>
                </tr>"""

        html_content += f"""
            </table>
            </div>
            <script>
                // Pie Chart: Correct vs Incorrect
                new Chart(document.getElementById('resultsPieChart').getContext('2d'), {{
                    type: 'pie',
                    data: {{
                        labels: ['Correct', 'Incorrect'],
                        datasets: [{{ data: [{correct_predictions}, {incorrect_predictions}], backgroundColor: ['#28a745', '#dc3545'], borderWidth: 0 }}]
                    }},
                    options: {{ plugins: {{ legend: {{ display: true, position: 'bottom' }}, title: {{ display: true, text: 'Prediction Accuracy ({accuracy:.2f}%)' }} }} }}
                }});

                // Line Chart: Rolling Accuracy
                new Chart(document.getElementById('accuracyLineChart').getContext('2d'), {{
                    type: 'line',
                    data: {{
                        labels: {rolling_accuracy_labels},
                        datasets: [{{
                            label: 'Rolling Accuracy (%)', data: {rolling_accuracy_data}, borderColor: '#007bff',
                            backgroundColor: 'rgba(0, 123, 255, 0.1)', fill: true, tension: 0.1, pointRadius: 0
                        }}]
                    }},
                    options: {{
                        scales: {{ y: {{ beginAtZero: true, max: 100, ticks: {{ callback: v => v + '%' }} }}, x: {{ ticks: {{ maxRotation: 0, autoSkip: true, autoSkipPadding: 20 }} }} }},
                        plugins: {{ legend: {{ display: false }}, title: {{ display: true, text: '{rolling_accuracy_chart_title}' }} }}
                    }}
                }});
                // Removed loss streak bar chart
            </script>
        </body>
        </html>
        """

        # Create Output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "backtest_results.html")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"\nBacktest complete. Results saved to {output_file}")
        print(f"Accuracy: {accuracy:.2f}% (MCC: {mcc:.3f})")
