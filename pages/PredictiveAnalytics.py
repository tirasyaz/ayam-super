import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, mean_squared_error
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

# Check if the shapes of the prediction and true labels match
st.write(f"y_test_class shape: {y_test_class.shape}")
st.write(f"y_pred_class shape: {y_pred_class.shape}")

# Ensure the lengths match before proceeding
if len(y_test_class) == len(y_pred_class):
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

    # Reintegrate 'item_code' for mapping predictions
    X_test_with_item_code = X_test.reshape(X_test.shape[0], -1)  # Reshape for compatibility
    X_test_with_item_code = pd.DataFrame(X_test_with_item_code, columns=X.columns)
    X_test_with_item_code['item_code'] = data.loc[X_test_with_item_code.index, 'item_code'].values

    # Map predictions back to original scale
    if len(y_test.shape) > 1:  # Multi-class case
        y_pred_prices = np.argmax(y_pred, axis=1)
    else:
        y_pred_prices = y_pred.reshape(-1)

    # Ensure both arrays have the same length before combining
    if len(X_test_with_item_code) == len(y_pred_prices):
        predictions = pd.DataFrame({
            'item_code': X_test_with_item_code['item_code'].values,  # Use .values to ensure correct alignment
            'predicted_price': y_pred_prices
        }).reset_index(drop=True)
    else:
        st.error("Error: Length mismatch between item_code and predicted prices.")

    # Filter predictions for item codes 1, 2, and 3
    filtered_predictions = predictions[predictions['item_code'].isin([1, 2, 3])]

    # Display predictions for item codes 1, 2, and 3
    st.subheader('Predicted Prices for Item Codes 1, 2, and 3')
    st.write(filtered_predictions)

    # Visualize the predicted prices
    st.subheader('Predicted Prices Visualization')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=filtered_predictions,
        x='item_code',
        y='predicted_price',
        palette='viridis'
    )
    plt.title('Predicted Prices for Item Codes 1, 2 and 3')
    plt.xlabel('Item Code')
    plt.ylabel('Predicted Price')
    plt.xticks([0, 1, 2], ['Item 1', 'Item 2', 'Item 3'])
    plt.tight_layout()
    st.pyplot(fig)

    # Visualize training and validation accuracy
    st.subheader('Training and Validation Accuracy')
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    st.pyplot(fig)

else:
    st.error("Shape mismatch between y_test_class and y_pred_class")
