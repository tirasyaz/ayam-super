import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Title of the app
st.title("Predictive Analytics for Ayam Prices with XGBoost")

# Load your dataset
file_path = 'https://raw.githubusercontent.com/tirasyaz/ayam-super/refs/heads/main/filtered_pricecatcher_data.csv'
st.write("Loading dataset...")
data = pd.read_csv(file_path)

# Preprocessing: Handle missing values and encode categorical data
st.write("Preprocessing the data...")
data = data.dropna()
data = pd.get_dummies(data, drop_first=True)

# Define features (X) and target (y)
X = data.drop(columns=['price'])  # Drop target column
y = data['price']  # Target column (assuming continuous prices)

# Normalize features (XGBoost works fine without deep scaling, but it's still good practice)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Use XGBRegressor for regression task
model = xgb.XGBRegressor(objective='reg:squarederror')

# Train the XGBoost model
st.write("Training the XGBoost model...")
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error (MSE): {mse:.2f}")

# Reintegrate 'item_code' for mapping predictions
X_test_with_item_code = pd.DataFrame(X_test, columns=X.columns)
X_test_with_item_code['item_code'] = data.loc[X_test_with_item_code.index, 'item_code'].values

# Map predictions back to original scale
predictions = pd.DataFrame({
    'item_code': X_test_with_item_code['item_code'],
    'predicted_price': y_pred
})

# Filter predictions for item codes 1, 2, and 3
filtered_predictions = predictions[predictions['item_code'].isin([1, 2, 3])]

# Display predictions for item codes 1, 2, and 3
st.write("Predicted Prices for Item Codes 1, 2, and 3:")
st.write(filtered_predictions)

# Visualize the predicted prices
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

# Scatter plot of Actual vs. Predicted prices
st.write("Scatter Plot of Actual vs Predicted Prices:")
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.scatter(y_test, y_pred, alpha=0.6, color='b')
ax2.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
ax2.set_title('Actual vs Predicted Prices')
ax2.set_xlabel('Actual Prices')
ax2.set_ylabel('Predicted Prices')
ax2.grid(True)
st.pyplot(fig2)
