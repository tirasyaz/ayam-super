import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import numpy as np
import streamlit as st

# Streamlit app title
st.title('Price Forecasting with ARIMA')

# Load your dataset
data_url = 'https://raw.githubusercontent.com/tirasyaz/ayam-super/refs/heads/main/filtered_pricecatcher_data.csv'
data = pd.read_csv(data_url)

# Ensure the date column is in datetime format
data['date'] = pd.to_datetime(data['date'])

# List of item codes to predict
item_codes = [1, 2, 3]

# User selects the item code
selected_code = st.selectbox("Select item code to forecast:", item_codes)

# Filter data for the selected item_code
item_data = data[data['item_code'] == selected_code]

if item_data.empty:
    st.warning(f"No data found for item_code {selected_code}.")
else:
    # Set the date as the index and sort by date
    item_data = item_data.set_index('date').sort_index()

    # Aggregate data by weekly average for better clarity
    item_data = item_data[['price']].resample('W').mean()

    # Extract the price column
    price_data = item_data['price']

    # Display the raw data
    st.subheader(f"Raw Data for item_code {selected_code}")
    st.dataframe(item_data)

    # Check for stationarity
    adf_test = adfuller(price_data.dropna())
    st.write(f"ADF Statistic: {adf_test[0]}")
    st.write(f"p-value: {adf_test[1]}")

    # Difference the data if non-stationary
    if adf_test[1] > 0.05:
        price_data_diff = price_data.diff().dropna()
        st.write("Data was differenced to achieve stationarity.")
    else:
        price_data_diff = price_data
        st.write("Data is already stationary.")

    # Fit ARIMA model
    model = ARIMA(price_data, order=(1, 1, 1))  # Example ARIMA(1, 1, 1)
    model_fit = model.fit()

    # Forecast future prices
    forecast_steps = 30  # Predict for the next 30 days
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(price_data.index[-1], periods=forecast_steps + 1, freq='W')[1:]
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # Evaluate model using RMSE
    train_size = int(len(price_data) * 0.8)
    train, test = price_data[:train_size], price_data[train_size:]
    model = ARIMA(train, order=(1, 1, 1))
    model_fit = model.fit()
    forecast_test = model_fit.forecast(steps=len(test))

    test = test.dropna()
    forecast_test = forecast_test[test.index]  # Align forecast with test index

    rmse = np.sqrt(mean_squared_error(test, forecast_test))
    st.write(f"Test RMSE for item_code {selected_code}: {rmse:.2f}")

    # Visualization
    st.subheader(f"Forecast Visualization for item_code {selected_code}")

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(price_data, label="Observed Prices", alpha=0.7)
    ax.plot(
        forecast_index,
        forecast_mean,
        label="Forecasted Prices",
        linestyle="--",
        color="orange",
    )
    ax.fill_between(
        forecast_index,
        forecast_ci.iloc[:, 0],
        forecast_ci.iloc[:, 1],
        color="orange",
        alpha=0.2,
        label="Confidence Interval",
    )
    ax.set_title("Price Forecast (Weekly Averaged Data)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)

    # Display the plot
    st.pyplot(fig)
