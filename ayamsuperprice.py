import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import requests
from io import StringIO

# Streamlit app title
st.title("Negeri Sembilan Population and Price Analysis")

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
st.header("Data Loading")
st.write("Loading filtered datasets for analysis...")

population_data = load_data_from_github(population_file_url)
price_data = load_data_from_github(price_file_url)

st.success("Data loaded successfully!")

# Interactive Filter Options
st.sidebar.header("Filter Options")
district_filter = st.sidebar.selectbox("Select District", population_data['district'].unique())
item_code_filter = st.sidebar.selectbox("Select Item Code", price_data['item_code'].unique())

# Population Analysis
st.header("Population by District")
if 'district' in population_data.columns and 'population' in population_data.columns:
    # Filter population data based on district selection
    filtered_population = population_data[population_data['district'] == district_filter]

    # Group by district and calculate the total population
    district_population = (
        filtered_population.groupby('district')['population'].sum().reset_index()
        .sort_values(by='population', ascending=False)
    )

    # Display data table
    st.subheader("Population Data Table")
    st.dataframe(district_population)

    # Interactive Plotly Bar Chart for population
    st.subheader("Population Distribution")
    fig = px.bar(district_population, x='district', y='population',
                 title="Population by District in Negeri Sembilan",
                 labels={'district': 'District', 'population': 'Population'},
                 color='district', template="plotly_dark")
    st.plotly_chart(fig)
else:
    st.warning("Required columns 'district' and 'population' are not present in the population data.")

# Price Analysis
st.header("Price Analysis")
if {'district', 'item_code', 'price', 'premise_type'}.issubset(price_data.columns):
    # Filter price data based on item code selection
    filtered_price = price_data[price_data['item_code'] == item_code_filter]

    # Summary Table: Group by district, item_code, and premise_type
    st.subheader("Summary Table")
    summary_table = filtered_price.groupby(['district', 'item_code', 'premise_type']).agg(
        count=('price', 'count'),
        avg_price=('price', 'mean')
    ).reset_index()
    st.dataframe(summary_table)

    # Interactive Plotly Bar Chart for average price by district and item code
    st.subheader("Average Price by District and Item Code")
    fig = px.bar(filtered_price, x='district', y='price', color='item_code',
                 title="Average Price by District and Item Code",
                 labels={'district': 'District', 'price': 'Price'},
                 template="plotly_dark", barmode='group')
    st.plotly_chart(fig)

else:
    st.warning("Required columns for price analysis are not present in the dataset.")
