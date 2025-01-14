import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset from GitHub
file_path = 'https://raw.githubusercontent.com/tirasyaz/ayam-super/refs/heads/main/filtered_pricecatcher_data.csv'
data = pd.read_csv(file_path)

# Streamlit Title and Description
st.title('Predictive Analytics with LSTM')
st.write('This app uses LSTM to predict item prices based on features in the dataset.')

# Preprocessing: Handle missing values and encode categorical data
data = data.dropna()
data = pd.get_dummies(data, drop_first=True)

# Define features (X) and target (y)
X = data.drop(columns=['price'])  # Drop target column
y = data['price']  # Target column

# Normalize features (LSTMs require scaled data for efficient training)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape data for LSTM: (samples, timesteps, features)
X_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y, test_size=0.2, random_state=42)

# Convert target to categorical if classification is multi-class
if len(np.unique(y)) > 2:
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

# Build the LSTM model
model = Sequential([
    LSTM(20, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(20, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(y_train.shape[1] if len(y_train.shape) > 1 else 1, activation='softmax' if len(y_train.shape) > 1 else 'sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy' if len(y_train.shape) > 1 else 'binary_crossentropy', metrics=['accuracy'])

# Train the model with 20 epochs
st.write('Training the model...')
history = model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32, verbose=1)

# Predict and evaluate
y_pred = model.predict(X_test)
if len(y_test.shape) > 1:  # Multi-class case
    y_pred_class = np.argmax(y_pred, axis=1)
    y_test_class = np.argmax(y_test, axis=1)
else:  # Binary classification case
    y_pred_class = (y_pred > 0.5).astype(int).reshape(-1)
    y_test_class = y_test

# Evaluation metrics
accuracy = accuracy_score(y_test_class, y_pred_class)
precision = precision_score(y_test_class, y_pred_class, average='weighted')
recall = recall_score(y_test_class, y_pred_class, average='weighted')
f1 = f1_score(y_test_class, y_pred_class, average='weighted')
mse = mean_squared_error(y_test_class, y_pred_class)

# Display evaluation metrics
st.subheader('Model Evaluation Metrics')
st.write(f"**Accuracy:** {accuracy:.2f}")
st.write(f"**Precision:** {precision:.2f}")
st.write(f"**Recall:** {recall:.2f}")
st.write(f"**F1-Score:** {f1:.2f}")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_test_class, y_pred_class)
st.subheader('Confusion Matrix')
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Failure', 'Failure'], yticklabels=['No Failure', 'Failure'])
plt.xlabel('Predicted')
plt.ylabel('True')
st.pyplot(fig)

st.subheader('Classification Report')
st.text(classification_report(y_test_class, y_pred_class))

# Prepare monthly prices for visualization
data['date'] = pd.to_datetime(data['date'])
data['month_year'] = data['date'].dt.to_period('M')
monthly_prices = data.groupby(['month_year', 'item_code'])['price'].mean().reset_index()

# Store predicted values and corresponding dates
all_predictions = []

# Predict for each item
for item in data['item_code'].unique():
    item_data = data[data['item_code'] == item]['price'].values
    item_predictions = model.predict(item_data)  # Predict using the LSTM model
    all_predictions.append((item, item_predictions))

# Visualization of actual and predicted prices
st.subheader('Monthly Prices for Each Item with Predictions')
fig, ax = plt.subplots(figsize=(15, 6))
for item in monthly_prices['item_code'].unique():
    # Plot actual prices
    item_data = monthly_prices[monthly_prices['item_code'] == item]
    ax.plot(item_data['month_year'].astype(str), item_data['price'], label=f'Actual Prices - Item {item}')
    
    # Plot predicted prices
    # Retrieve predictions and future dates for the current item
    predictions, future_dates = next(((pred, dates) for itm, pred, dates in all_predictions if itm == item), (None, None))
    
    if predictions is not None and future_dates is not None:
        ax.plot(future_dates, predictions.flatten(), linestyle='--', label=f'Predicted Prices - Item {item}')

ax.legend()
ax.set_title('Monthly Prices for Each Item with Predictions')
ax.set_xlabel('Month-Year')
ax.set_ylabel('Average Price')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)
