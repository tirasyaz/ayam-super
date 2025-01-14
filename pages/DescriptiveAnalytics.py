import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load dataset from GitHub
file_path = 'https://raw.githubusercontent.com/tirasyaz/ayam-super/refs/heads/main/filtered_pricecatcher_data.csv'
data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data['date'])

# Display DataFrame in the app
st.write("### Data Preview", data.head())

# Descriptive statistics for numerical columns (e.g., price)
summary_statistics = {
    "Mean": data['price'].mean(),
    "Median": data['price'].median(),
    "Mode": data['price'].mode()[0],
    "Range": data['price'].max() - data['price'].min(),
    "Standard Deviation": data['price'].std()
}

# Display the descriptive statistics in the app
st.write("### Descriptive Statistics for Price", summary_statistics)

# Unique counts for categorical columns
categorical_summary = {
    "Premise Types": data['premise_type'].nunique(),
    "States": data['state'].nunique(),
    "Districts": data['district'].nunique(),
    "Items": data['item_code'].nunique(),
    "Premises": data['premise_code'].nunique()
}

st.write("### Categorical Summary", categorical_summary)

# Price distribution histogram
st.write("### Price Distribution Histogram")
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data['price'], bins=20, kde=True, color='blue', ax=ax)
ax.set(title="Price Distribution", xlabel="Price", ylabel="Frequency")
st.pyplot(fig)

# Average price by district
st.write("### Average Price by District")
fig, ax = plt.subplots(figsize=(10, 6))
avg_price_district = data.groupby('district')['price'].mean().sort_values()
sns.barplot(x=avg_price_district.values, y=avg_price_district.index, palette='viridis', ax=ax)
ax.set(title="Average Price by District", xlabel="Average Price", ylabel="District")
st.pyplot(fig)

# Time-series plot for price trends (overall)
st.write("### Price Trend Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
avg_price_date = data.groupby('date')['price'].mean()
sns.lineplot(x=avg_price_date.index, y=avg_price_date.values, color='green', marker='o', ax=ax)
ax.set(title="Price Trend Over Time", xlabel="Date", ylabel="Average Price")
plt.xticks(rotation=45)
st.pyplot(fig)
