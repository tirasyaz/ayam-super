import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import numpy as np
import streamlit as st

# Streamlit app title
st.title("ARIMA Price Forecast Visualization")

# Cache data loading function
@st.cache_data
def load_data():
    return pd.read_csv('https://raw.githubusercontent.com/tirasyaz/ayam-super/refs/heads/main/filtered_pricecatcher_data.csv')

# Cache the ARIMA model fitting function
@st.cache_resource
def fit_arima_model(price_data, order=(1, 1, 1)):
    model = ARIMA(price_data, order=order)
    model_fit = model.fit()
    return model_fit

# Load your dataset
data = load_data()

# Ensure the date column is in datetime format
data['date'] = pd.to_datetime(data['date'])

# Sidebar for user inputs
st.sidebar.header("Settings")
item_codes = st.sidebar.multiselect(
    "Select Item Codes to Analyze", options=data['item_code'].unique(), default=[1, 2, 3]
)
forecast_steps = st.sidebar.slider("Forecast Steps (Days)", min_value=7, max_value=90, value=30, step=7)

# Initialize variables
forecast_results = []
overall_rmse = []

# Loop through each selected item_code
for code in item_codes:
    st.subheader(f"Processing item_code: {code}")

    # Filter data for the current item_code
    item_data = data[data['item_code'] == code]

    # Set the date as the index and sort by date
    item_data = item_data.set_index('date').sort_index()

    # Aggregate data by weekly average for better clarity
    item_data = item_data[['price']].resample('W').mean()

    # Extract the price column
    price_data = item_data['price']

    # Check for stationarity and difference the data if necessary
    adf_test = adfuller(price_data.dropna())
    st.write(f"ADF Statistic for item_code {code}: {adf_test[0]}")
    st.write(f"p-value for item_code {code}: {adf_test[1]}")

    # Difference the data if non-stationary
    if adf_test[1] > 0.05:
        price_data_diff = price_data.diff().dropna()
    else:
        price_data_diff = price_data

    # Fit ARIMA model and get forecast
    model_fit = fit_arima_model(price_data_diff)
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(price_data.index[-1], periods=forecast_steps + 1, freq='W')[1:]
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # Store results for visualization
    forecast_results.append({
        'item_code': code,
        'observed': price_data,
        'forecast_index': forecast_index,
        'forecast_mean': forecast_mean,
        'forecast_ci': forecast_ci,
    })

    # Evaluate model using RMSE
    train_size = int(len(price_data) * 0.8)
    train, test = price_data[:train_size], price_data[train_size:]
    model_fit = fit_arima_model(train)
    forecast_test = model_fit.forecast(steps=len(test))

    test = test.dropna()
    forecast_test = forecast_test[test.index]  # Align forecast with test index

    rmse = np.sqrt(mean_squared_error(test, forecast_test))
    overall_rmse.append(rmse)
    st.write(f"Test RMSE for item_code {code}: {rmse}")

# Compute overall RMSE
average_rmse = np.mean(overall_rmse)
st.write("Overall RMSE:", average_rmse)

# Visualization
st.subheader("Price Forecast for All Selected Item Codes")
for result in forecast_results:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result['observed'], label=f'Observed (item_code {result["item_code"]})', alpha=0.7, color='blue')
    ax.plot(result['forecast_index'], result['forecast_mean'], label='Forecast', linestyle='--', color='orange')
    ax.fill_between(
        result['forecast_index'],
        result['forecast_ci'].iloc[:, 0],
        result['forecast_ci'].iloc[:, 1],
        alpha=0.2,
        color='orange',
        label='95% Confidence Interval'
    )
    ax.set_title(f"Price Forecast for item_code {result['item_code']} (Weekly Averaged Data)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
