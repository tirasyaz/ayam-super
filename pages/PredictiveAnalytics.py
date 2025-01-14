import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import numpy as np

# Load dataset from GitHub
file_path = 'https://raw.githubusercontent.com/tirasyaz/ayam-super/refs/heads/main/filtered_pricecatcher_data.csv'
data = pd.read_csv(file_path)

# Ensure the date column is in datetime format
data['date'] = pd.to_datetime(data['date'])

# Initialize variables
overall_rmse = []
forecast_results = []

# List of item codes to predict
item_codes = [1, 2, 3]

# Loop through each item_code
for code in item_codes:
    print(f"\nProcessing item_code: {code}")

    # Filter data for the current item_code
    item_data = data[data['item_code'] == code]

    # Set the date as the index and sort by date
    item_data = item_data.set_index('date').sort_index()

    # Aggregate data by weekly average for better clarity
    item_data = item_data[['price']].resample('W').mean()

    # Extract the price column
    price_data = item_data['price']

    # Check for stationarity
    adf_test = adfuller(price_data.dropna())
    print(f'ADF Statistic for item_code {code}:', adf_test[0])
    print(f'p-value for item_code {code}:', adf_test[1])

    # Difference the data if non-stationary
    if adf_test[1] > 0.05:
        price_data_diff = price_data.diff().dropna()
    else:
        price_data_diff = price_data

    # Fit ARIMA model
    model = ARIMA(price_data, order=(1, 1, 1))  # Example ARIMA(1, 1, 1)
    model_fit = model.fit()

    # Forecast future prices
    forecast_steps = 30  # Predict for the next 30 weeks
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
    model = ARIMA(train, order=(1, 1, 1))
    model_fit = model.fit()
    forecast_test = model_fit.forecast(steps=len(test))

    test = test.dropna()
    forecast_test = forecast_test[:len(test)]  # Align forecast with test index

    rmse = np.sqrt(mean_squared_error(test, forecast_test))
    overall_rmse.append(rmse)
    print(f'Test RMSE for item_code {code}:', rmse)

# Compute overall RMSE
average_rmse = np.mean(overall_rmse)
print("\nOverall RMSE:", average_rmse)

# Visualization of all forecasts combined
plt.figure(figsize=(14, 8))

for result in forecast_results:
    plt.plot(
        result['observed'],
        label=f'Observed (item_code {result["item_code"]})',
        alpha=0.7,
    )
    plt.plot(
        result['forecast_index'],
        result['forecast_mean'],
        label=f'Forecast (item_code {result["item_code"]})',
        linestyle='--',
    )

    # Add confidence interval shading
    plt.fill_between(
        result['forecast_index'],
        result['forecast_ci'].iloc[:, 0],
        result['forecast_ci'].iloc[:, 1],
        alpha=0.2,
    )

plt.title("Price Forecast for All Item Codes (Weekly Averaged Data)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
