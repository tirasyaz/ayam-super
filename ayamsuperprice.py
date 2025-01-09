import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO

# Streamlit app title
st.title("Negeri Sembilan Population and Price Analysis")

# GitHub raw file URLs
population_file_url = 'https://raw.githubusercontent.com/tirasyaz/ayam-super/refs/heads/main/filtered_population_district.csv'
price_file_url = 'https://raw.githubusercontent.com/tirasyaz/repository-name/main/Filtered/filtered_pricecatcher_data.csv'

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

# Population Analysis
st.header("Population by District")
if 'district' in population_data.columns and 'population' in population_data.columns:
    # Group by district and calculate the total population
    district_population = (
        population_data.groupby('district')['population'].sum().reset_index()
        .sort_values(by='population', ascending=False)
    )

    # Display data table
    st.subheader("Population Data Table")
    st.dataframe(district_population)

    # Plot population data
    st.subheader("Population Distribution")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(district_population['district'], district_population['population'], color='skyblue')
    ax.set_xlabel('District')
    ax.set_ylabel('Population')
    ax.set_title('Population by District in Negeri Sembilan')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
else:
    st.warning("Required columns 'district' and 'population' are not present in the population data.")

# Price Analysis
st.header("Price Analysis")
if {'district', 'item_code', 'price', 'premise_type'}.issubset(price_data.columns):
    # Summary Table: Group by district, item_code, and premise_type
    st.subheader("Summary Table")
    summary_table = price_data.groupby(['district', 'item_code', 'premise_type']).agg(
        count=('price', 'count'),
        avg_price=('price', 'mean')
    ).reset_index()
    st.dataframe(summary_table)

    # Average price by district and item_code
    avg_price_data = price_data.groupby(['district', 'item_code']).agg(
        avg_price=('price', 'mean')
    ).reset_index()

    # Bar plot for average price
    st.subheader("Average Price by District and Item Code")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=avg_price_data,
        x='district',
        y='avg_price',
        hue='item_code',
        palette='viridis',
        ax=ax
    )
    ax.set_title('Average Price by District and Item Code')
    ax.set_xlabel('District')
    ax.set_ylabel('Average Price')
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.warning("Required columns for price analysis are not present in the dataset.")
