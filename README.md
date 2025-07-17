# BTC Up/Down Algorithm Template

Welcome! This project lets you **create and test your own Bitcoin price prediction algorithms**—no coding experience required.

## 🚀 Quick Start

1. **Install Python**  
   Download and install Python 3.8 or newer from [python.org](https://www.python.org/downloads/).

2. **Download this Project**  
   - Click the green "Code" button above and choose "Download ZIP".
   - Unzip the folder to your computer.

3. **Setup the Environment**
   ```
   # Create a virtual environment
   python -m venv venv
   
   # Activate the environment
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

4. **Add Historical Data**  
   - Place your historical price data CSV file in the `data` folder.
   - The file should look like:  
     `timestamp,open,high,low,close,volume`  
     Example row:  
     `2024-01-01 00:00:00,42000,42100,41900,42050,1.23`

5. **Configure Settings**  
   - Open `config/config.py` and set the filename for your data (e.g., `BACKTEST_FILE = "data/your_data.csv"`).
   - Adjust other settings if you want.

6. **Run the Backtester**  
   Open a terminal in this folder and run:
   ```
   python main.py
   ```
   Then choose "backtest" when prompted.
   - This will test your algorithm and create a file called `backtest_results.html` with your results.

## 🧠 How to Make Your Own Algorithm

1. **Open `src/models/predictor.py`**  
   - Find the `make_prediction()` method.
   - Change the logic to use your own rules!
   - Example: Use moving averages, RSI, or any idea you have.

2. **How to Predict**
   - Return `1` for "UP", `0` for "DOWN".
   - Return a confidence score (0.0 to 1.0).
   - Return a short explanation string.

   Example:
   ```python
   def make_prediction(self, current_price, timestamp):
       if current_price > 40000:
           return 1, 0.9, "Price above 40k, predicting UP"
       else:
           return 0, 0.7, "Price below 40k, predicting DOWN"
   ```

3. **Test Again!**  
   - Save your changes.
   - Run `python main.py` again to see your results.

## 📊 Understanding Results

- After running, open `Output/backtest_results.html` in your browser.
- See your prediction accuracy, streaks, and detailed results.

## 🛠️ Project Structure

```
btcUP_DOWN/
├── config/             # Configuration settings
├── data/               # Place your CSV data files here
├── Output/             # Generated reports and results
├── src/
│   ├── data/           # Data handling components
│   ├── models/         # Prediction algorithms
│   └── utils/          # Helper functions
├── tools/              # Testing and analysis tools
├── main.py             # Main entry point
└── requirements.txt    # Dependencies
```

## ❓ Need Help?

- If you get errors, check your CSV file format and settings in `config/config.py`.
- For more ideas, search for "Python trading indicators" or ask ChatGPT for help!

---

Happy coding and good luck with your predictions!
