import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Configuration
DATA_FILE_PATH = "/Data/Hourly/btc_indicators_data_full.csv"

# --- 1. Load Data ---
print("Loading data...")
try:
    df = pd.read_csv(DATA_FILE_PATH, parse_dates=['timestamp'])
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: Make sure '{DATA_FILE_PATH}' exists.")
    print("Please create this file from your prepared dataset.")
    print("The script expects columns like: timestamp, open, high, low, close, volume, indicators..., Target")
    exit()

# --- 2. Prepare Data for Modeling ---
print("Preparing data...")
# Define features (X) and target (y)
feature_columns = ['open', 'high', 'low', 'close', 'volume',
                   'MA_5', 'MACD', 'MACD_Signal', 'MACD_Hist',
                   'RSI_14', 'ATR_14', 'Volume_Change']
target_column = 'Target'

# Check if all feature columns exist in the dataframe
if not all(col in df.columns for col in feature_columns):
    print("Error: Not all feature columns are present in the training data.")
    print(f"Expected columns: {feature_columns}")
    print(f"Available columns: {list(df.columns)}")
    exit()

X = df[feature_columns]
y = df[target_column]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

# --- 3. Preprocessing: Scale Features ---
print("Scaling features...")
scaler = StandardScaler()
# Fit on training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. Train Logistic Regression Model ---
print("Training the Logistic Regression model...")
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
print("Model training complete.")

# --- 5. Evaluate Model ---
print("Evaluating the model...")
# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Model Performance ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Display a detailed classification report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Down (0)', 'Up (1)']))

# Display the confusion matrix
print("\n--- Confusion Matrix ---")
print("          Predicted")
print("         Down   Up")
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Actual Down: {conf_matrix[0]}")
print(f"Actual Up:   {conf_matrix[1]}")
print("-------------------------\n")
