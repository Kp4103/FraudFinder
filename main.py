import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import joblib
from keras.models import load_model

# Load and preprocess data
def preprocess_data(df):
    # Define the columns to be processed
    columns = ['Avg min between sent tnx', 'Avg min between received tnx',
               'Time Diff between first and last (Mins)',
               'Unique Received From Addresses', 'min value received',
               'max value received', 'avg val received', 'min val sent',
               'avg val sent', 'total transactions (including tnx to create contract',
               'total ether received', 'total ether balance']

    # Ensure that the log transformation is handled properly
    for c in columns:
        df[c] = df[c].apply(lambda x: np.log(x + 1) if x > 0 else 0)  # Adding 1 to avoid log(0)

    return df[columns]

# Streamlit UI
st.title("FraudFinder: Your Ethereum is your Ethereum.")

# Hardcoded paths to models
model_paths = {
    "Logistic Regression": 'logistic_regression_model.joblib',
    "Multilayer Perceptron": 'multi_layer_perceptron_model.h5',
    "Random Forest": 'random_forest_model.joblib',
}

# Model selection
selected_model = st.selectbox("Select a Model", list(model_paths.keys()))

# Load the pre-trained model
model = None

if selected_model == "Logistic Regression":
    try:
        model = joblib.load(model_paths[selected_model])
    except Exception as e:
        st.error(f"Error loading Logistic Regression model: {e}")
elif selected_model == "Multilayer Perceptron":
    try:
        model = load_model(model_paths[selected_model])
    except Exception as e:
        st.error(f"Error loading Multilayer Perceptron model: {e}")
elif selected_model == "Random Forest":
    try:
        model = joblib.load(model_paths[selected_model])
    except Exception as e:
        st.error(f"Error loading Random Forest model: {e}")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file for predictions", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Trim whitespace from column names
    df.columns = df.columns.str.strip()

    # Check if necessary columns exist
    required_columns = ['Avg min between sent tnx', 'Avg min between received tnx',
                        'Time Diff between first and last (Mins)',
                        'Unique Received From Addresses', 'min value received',
                        'max value received', 'avg val received', 'min val sent',
                        'avg val sent', 'total transactions (including tnx to create contract',
                        'total ether received', 'total ether balance']

    if not all(col in df.columns for col in required_columns):
        st.error("Uploaded CSV does not contain all the required columns.")
    else:
        # Preprocess the data
        processed_data = preprocess_data(df)

        # Make predictions on the uploaded data based on the selected model
        if model is not None:
            if selected_model == "Multilayer Perceptron":
                # For MLP, convert probabilities to binary predictions
                predictions = (model.predict(processed_data) > 0.5).astype(int)  # Assuming binary classification
            else:
                # For Logistic Regression and Random Forest
                predictions = model.predict(processed_data)

            # Add predictions to the DataFrame
            df['Predicted'] = predictions

            # Display rows identified as fraud
            fraud_cases = df[df['Predicted'] == 1]
            st.write("Detected Fraud Cases:")
            st.write(fraud_cases)

            # Check for true labels for confusion matrix
            if 'FLAG' in df.columns:
                y_true = df['FLAG']
                conf_matrix = confusion_matrix(y_true, predictions)
                st.write("Confusion Matrix:")
                st.write(conf_matrix)

                st.write("Classification Report:")
                class_report = classification_report(y_true, predictions)
                st.text(class_report)
            else:
                st.warning("No true labels available for calculating confusion matrix and classification report.")
        else:
            st.error("Model could not be loaded. Please check the selected model.")
