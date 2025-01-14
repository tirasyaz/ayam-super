import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Title and Introduction
st.title("XGBoost Regression for Ayam Prices")
st.markdown("This app predicts ayam prices using XGBoost and visualizes the results.")

# Load dataset
st.write("Loading dataset...")
file_path = 'https://raw.githubusercontent.com/tirasyaz/ayam-super/refs/heads/main/filtered_pricecatcher_data.csv'

try:
    data = pd.read_csv(file_path)
    st.write("Dataset Loaded Successfully!")
    st.write(data.head())  # Display the first few rows
except Exception as e:
    st.error("Error loading dataset. Please check the file path or URL.")
    st.stop()

# Preprocessing
st.write("Preprocessing the data...")
data = data.dropna()
data = pd.get_dummies(data, drop_first=True)

# Feature and Target Separation
X = data.drop(columns=['price'])  # Drop target column
y = data['price']  # Target column

# Normalize Features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train XGBoost Model
st.write("Training the XGBoost Model...")
model = xgb.XGBRegressor(objective='reg:squarederror')

try:
    model.fit(X_train, y_train)
    st.write("Model trained successfully!")
except Exception as e:
    st.error("Error during model training.")
    st.stop()

# Predictions and Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error (MSE): {mse:.2f}")

# Reintegrate 'item_code' for Mapping Predictions
try:
    X_test_with_item_code = pd.DataFrame(X_test, columns=X.columns)
    X_test_with_item_code['item_code'] = data.loc[X_test_with_item_code.index, 'item_code'].values

    predictions = pd.DataFrame({
        'item_code': X_test_with_item_code['item_code'],
        'predicted_price': y_pred
    })
except Exception as e:
    st.error("Error during reintegration of item codes.")
    st.stop()

# Filter Predictions
filtered_predictions = predictions[predictions['item_code'].isin([1, 2, 3])]
st.write("Predicted Prices for Item Codes 1, 2, and 3:")
st.write(filtered_predictions)

# Visualization: Bar Chart
st.write("Bar Chart of Predicted Prices for Item Codes 1, 2, and 3:")
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(
    data=filtered_predictions,
    x='item_code',
    y='predicted_price',
    palette='viridis',
    ax=ax
)
ax.set_title('Predicted Prices for Item Codes 1, 2, and 3')
ax.set_xlabel('Item Code')
ax.set_ylabel('Predicted Price')
ax.set_xticklabels(['Item 1', 'Item 2', 'Item 3'])
st.pyplot(fig)

# Visualization: Scatter Plot
st.write("Scatter Plot of Actual vs Predicted Prices:")
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.scatter(y_test, y_pred, alpha=0.6, color='b')
ax2.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
ax2.set_title('Actual vs Predicted Prices')
ax2.set_xlabel('Actual Prices')
ax2.set_ylabel('Predicted Prices')
ax2.grid(True)
st.pyplot(fig2)

