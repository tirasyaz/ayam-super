import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from GitHub
st.title("LSTM Predictive Price Analysis")
st.header("Load Dataset")
file_path = 'https://raw.githubusercontent.com/tirasyaz/ayam-super/refs/heads/main/filtered_pricecatcher_data.csv'

@st.cache_data
def load_data(url):
    return pd.read_csv(url)

data = load_data(file_path)
st.write("Loaded Dataset:")
st.dataframe(data.head())

# Preprocess the data
st.header("Preprocessing Data")
data = data.dropna()
data = pd.get_dummies(data, drop_first=True)

X = data.drop(columns=['price'])
y = data['price']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y, test_size=0.2, random_state=42)

# Convert target to categorical if necessary
if len(np.unique(y)) > 2:
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

# Model definition
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(y_train.shape[1] if len(y_train.shape) > 1 else 1, activation='softmax' if len(y_train.shape) > 1 else 'sigmoid')
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy' if len(y_train.shape) > 1 else 'binary_crossentropy', 
              metrics=['accuracy'])

# Train the model
st.header("Training the Model")
with st.spinner("Training the LSTM model..."):
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=0)
st.success("Model training complete!")

# Metrics and Predictions
st.header("Evaluation Metrics")
y_pred = model.predict(X_test)

if len(y_test.shape) > 1:  # Multi-class case
    y_pred_class = np.argmax(y_pred, axis=1)
    y_test_class = np.argmax(y_test, axis=1)
else:  # Binary classification case
    y_pred_class = (y_pred > 0.5).astype(int).reshape(-1)
    y_test_class = y_test

accuracy = accuracy_score(y_test_class, y_pred_class)
precision = precision_score(y_test_class, y_pred_class, average='weighted')
recall = recall_score(y_test_class, y_pred_class, average='weighted')
f1 = f1_score(y_test_class, y_pred_class, average='weighted')
mse = mean_squared_error(y_test_class, y_pred_class)

st.write(f"**Accuracy**: {accuracy:.2f}")
st.write(f"**Precision**: {precision:.2f}")
st.write(f"**Recall**: {recall:.2f}")
st.write(f"**F1-Score**: {f1:.2f}")
st.write(f"**Mean Squared Error (MSE)**: {mse:.2f}")

# Confusion matrix
st.subheader("Confusion Matrix")
conf_matrix = confusion_matrix(y_test_class, y_pred_class)
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Visualize training history
st.subheader("Training History")
fig, ax = plt.subplots()
ax.plot(history.history['accuracy'], label='Training Accuracy')
ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax.set_title("Model Accuracy")
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy")
ax.legend()
st.pyplot(fig)

# Predictions for specific items
st.header("Predictions for Item Codes 1, 2, and 3")
X_test_with_item_code = X_test.reshape(X_test.shape[0], -1)
X_test_with_item_code = pd.DataFrame(X_test_with_item_code, columns=X.columns)
X_test_with_item_code['item_code'] = data.loc[X_test_with_item_code.index, 'item_code'].values

y_pred_prices = y_pred.reshape(-1)
predictions = pd.DataFrame({
    'item_code': X_test_with_item_code['item_code'],
    'predicted_price': y_pred_prices
})

filtered_predictions = predictions[predictions['item_code'].isin([1, 2, 3])]
st.dataframe(filtered_predictions)

# Bar plot for predictions
st.subheader("Predicted Prices Visualization")
fig = px.bar(filtered_predictions, x='item_code', y='predicted_price', 
             title="Predicted Prices for Item Codes 1, 2, and 3", color='item_code')
st.plotly_chart(fig)

# Save predictions
output_path = 'predicted_prices_item_codes.csv'
filtered_predictions.to_csv(output_path, index=False)
st.success(f"Filtered predictions saved as `{output_path}`")
