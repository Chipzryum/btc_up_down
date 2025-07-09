import pandas as pd
import os
from config.config import BACKTEST_FILE, BACKTEST_TRIGGER_MINUTE, REQUIRED_MINUTES
from Models.predictor import PricePredictor

class Backtester:
    def __init__(self):
        self.predictor = PricePredictor()
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

                if len(group) < REQUIRED_MINUTES:
                    print(f"Skipping hour {current_hour}, not enough minutes in group.")
                    continue

                if BACKTEST_TRIGGER_MINUTE >= len(group):
                    print(f"Skipping hour {current_hour}, not enough data for trigger minute.")
                    continue

                trigger_candle = group.iloc[BACKTEST_TRIGGER_MINUTE]

                live_update = {
                    'timestamp': trigger_candle['timestamp'],
                    'price': trigger_candle['open']
                }

                prediction, confidence, explanation = self.predictor.process_live_update(live_update)

                if prediction is not None:
                    hourly_open = group.iloc[0]['open']
                    hourly_close = group.iloc[-1]['close']
                    actual_outcome = 1 if hourly_close > hourly_open else 0
                    correct = "✅" if prediction == actual_outcome else "❌"

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
                    <canvas id="pieChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="streakChart"></canvas>
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
                const pieCtx = document.getElementById('pieChart').getContext('2d');
                new Chart(pieCtx, {
                    type: 'pie',
                    data: {
                        labels: ['Correct', 'Incorrect'],
                        datasets: [{
                            data: [%d, %d],
                            backgroundColor: ['#4caf50', '#f44336'],
                        }]
                    },
                    options: {
                        plugins: {
                            legend: { display: true, position: 'bottom' },
                            title: { display: true, text: 'Prediction Results' }
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
                            data: [%d, %d],
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
            </script>
        </body>
        </html>
        """ % (correct_predictions, incorrect_predictions, longest_win_streak, longest_loss_streak)

        # Create Output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Output")
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "backtest_results.html")
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        print(f"\nBacktest complete. Results saved to {output_file}")
        print(f"Accuracy: {accuracy:.2f}% ({correct_predictions} correct out of {total_predictions})")
