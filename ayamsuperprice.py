import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Streamlit app title
st.title("Negeri Sembilan Population and Price Analysis")

# File paths
folder_path = '/content/drive/MyDrive/AyamSuper/Filtered'
population_file = '/content/drive/MyDrive/AyamSuper/Filtered/filtered_population_district.csv'
price_file = os.path.join(folder_path, 'filtered_pricecatcher_data.csv')

# Load datasets
st.header("Data Loading")
st.write("Loading filtered datasets for analysis...")

try:
    population_data = pd.read_csv(population_file)
    price_data = pd.read_csv(price_file)
    st.success("Data loaded successfully!")
except FileNotFoundError as e:
    st.error(f"Error loading data: {e}")
    st.stop()

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
