import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from io import StringIO

# Page title
st.title("Descriptive Analysis: Negeri Sembilan")

# GitHub raw file URLs
population_file_url = 'https://raw.githubusercontent.com/tirasyaz/ayam-super/refs/heads/main/filtered_population_district.csv'
price_file_url = 'https://raw.githubusercontent.com/tirasyaz/ayam-super/refs/heads/main/filtered_pricecatcher_data.csv'

# Function to load CSV from URL
def load_data_from_github(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        data = StringIO(response.text)
        return pd.read_csv(data)
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading data from GitHub: {e}")
        st.stop()

# Load datasets from GitHub
st.subheader("Data Loading")
population_data = load_data_from_github(population_file_url)
price_data = load_data_from_github(price_file_url)
st.success("Data loaded successfully!")

# Sidebar filters
st.sidebar.header("Filter Options")
district_filter = st.sidebar.selectbox("Select District", population_data['district'].unique())
item_code_filter = st.sidebar.selectbox("Select Item Code", price_data['item_code'].unique())

# Population Analysis
st.header("Population by District")
filtered_population = population_data[population_data['district'] == district_filter]
district_population = filtered_population.groupby('district')['population'].sum().reset_index()
st.subheader("Population Data Table")
st.dataframe(district_population)

fig = px.bar(district_population, x='district', y='population', 
             title="Population by District", color='district', template="plotly_dark")
st.plotly_chart(fig)

# Price Analysis
st.header("Price Analysis")
filtered_price = price_data[price_data['item_code'] == item_code_filter]
summary_table = filtered_price.groupby(['district', 'item_code', 'premise_type']).agg(
    count=('price', 'count'),
    avg_price=('price', 'mean')
).reset_index()
st.subheader("Summary Table")
st.dataframe(summary_table)

fig = px.bar(filtered_price, x='district', y='price', color='item_code',
             title="Average Price by District and Item Code", template="plotly_dark")
st.plotly_chart(fig)

fig = px.scatter(filtered_price, x='district', y='price', color='item_code',
                 title="Price Trend by District", template="plotly_dark")
st.plotly_chart(fig)
