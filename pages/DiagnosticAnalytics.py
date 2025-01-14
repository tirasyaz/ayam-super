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

# Correlation matrix
correlation = data.select_dtypes(include=['number']).corr()

# Display the correlation matrix
st.write("### Correlation Matrix")
st.dataframe(correlation)

# Pivot table: Average price by district and item_code
pivot_avg_price = data.pivot_table(values='price', index='district', columns='item_code', aggfunc='mean')

# Display pivot table
st.write("### Pivot Table: Average Price by District and Item Code")
st.dataframe(pivot_avg_price)

# Investigating variations by premise_type
avg_price_premise_type = data.groupby('premise_type')['price'].mean().sort_values()

# Display average prices by premise type
st.write("### Average Price by Premise Type:")
st.dataframe(avg_price_premise_type)

# Average price by district
avg_price_district = data.groupby('district')['price'].mean().sort_values()

# Visualizing average price by district
st.write("### Average Price by District")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=avg_price_district.values, y=avg_price_district.index, palette='viridis', ax=ax)
ax.set(title="Average Price by District", xlabel="Average Price", ylabel="District")
st.pyplot(fig)

# Visualizing average price by premise type
st.write("### Average Price by Premise Type")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=avg_price_premise_type.values, y=avg_price_premise_type.index, palette='coolwarm', ax=ax)
ax.set(title="Average Price by Premise Type", xlabel="Average Price", ylabel="Premise Type")
st.pyplot(fig)

# Average price over time
avg_price_time = data.groupby('date')['price'].mean()

# Visualizing price trend over time
st.write("### Price Trend Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x=avg_price_time.index, y=avg_price_time.values, color='blue', marker='o', ax=ax)
ax.set(title="Price Trend Over Time", xlabel="Date", ylabel="Average Price")
plt.xticks(rotation=45)
st.pyplot(fig)

# Average price by item code
avg_price_item = data.groupby('item_code')['price'].mean()

# Visualizing average price by item
st.write("### Average Price by Item Code")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=avg_price_item.index, y=avg_price_item.values, palette='muted', ax=ax)
ax.set(title="Average Price by Item Code", xlabel="Item Code", ylabel="Average Price")
st.pyplot(fig)

# Boxplot for price by district
st.write("### Price Distribution by District")
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(x='district', y='price', data=data, palette='Set2', ax=ax)
ax.set(title="Price Distribution by District", xlabel="District", ylabel="Price")
plt.xticks(rotation=45)
st.pyplot(fig)

# Boxplot for price by premise type
st.write("### Price Distribution by Premise Type")
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(x='premise_type', y='price', data=data, palette='Set3', ax=ax)
ax.set(title="Price Distribution by Premise Type", xlabel="Premise Type", ylabel="Price")
plt.xticks(rotation=45)
st.pyplot(fig)
