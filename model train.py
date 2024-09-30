import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\Users\kunal\Downloads\Compressed\Ethereum-Fraud-Detection-main\Ethereum-Fraud-Detection-main\Data\address_data_combined.csv"
df = pd.read_csv(file_path)

# Trim whitespace from column names
df.columns = df.columns.str.strip()

# Print the shape and the first few rows of the dataset
print(df.shape)
print(df.head())

# Check the number of fraud cases
f_txn = len(df[df['FLAG'] == 1])
print(f'Current amount of fraud is now {f_txn}, which is {f_txn / len(df) * 100:.2f}% of the original dataset')

# Drop rows with NaN values
df = df.dropna()
print(f'Shape after dropping NaN values: {df.shape}')

# Print the current columns in the DataFrame
print("Current columns in the DataFrame:")
print(df.columns.tolist())

# Dropping columns that have unique values < 5 (not useful features)
constant_columns = ['Unnamed: 0', 'Index', 'ERC20 uniq sent addr.1',
                    'ERC20 avg time between rec 2 tnx',
                    'ERC20 avg time between contract tnx',
                    'ERC20 min val sent contract',
                    'ERC20 max val sent contract',
                    'ERC20 avg val sent contract']

# Drop only those columns that are present in the DataFrame
columns_to_drop = [col for col in constant_columns if col in df.columns]
df.drop(columns=columns_to_drop, axis=1, inplace=True)
print(f'Shape after dropping constant columns: {df.shape}')

# Check for non-numeric columns and remove them
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric columns:", numeric_columns)

# Define target variable and features
y = df['FLAG']  # Adjust if your target variable is named differently
X = df[numeric_columns].drop(columns=['FLAG'], errors='ignore')  # Keep only numeric columns and drop the target

# Check if there are any remaining non-numeric columns
if X.shape[1] == 0:
    raise ValueError("No valid numeric features available for training the model.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model
model_filename = r"C:\Users\kunal\Downloads\random_forest_model.joblib"  # Update this path as needed
joblib.dump(rf_model, model_filename)
print(f'Model saved as {model_filename}')

# Evaluate the model
y_pred = rf_model.predict(X_test)
print(f'Test Accuracy: {np.mean(y_pred == y_test):.4f}')

# Print confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Plotting feature importances
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()
