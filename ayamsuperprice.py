import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the App
st.title("Negeri Sembilan Population and Price Analysis")

# File path or dynamic upload
DEFAULT_FILE_PATH = 'Filtered/filtered_population_district.csv'
uploaded_file = None

# Sidebar for file selection or upload
st.sidebar.title("File Options")
use_default = st.sidebar.checkbox("Use default file path", value=True)

if use_default:
    file_path = DEFAULT_FILE_PATH
else:
    uploaded_file = st.sidebar.file_uploader("Upload your filtered_population_district.csv file", type="csv")
    file_path = None

# Load the file based on user selection
try:
    if use_default and os.path.exists(file_path):
        df = pd.read_csv(file_path)
        st.success(f"Data loaded from default path: {file_path}")
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Data loaded from uploaded file.")
    else:
        st.warning("Please upload a file or provide a valid file path.")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Display the dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Basic Dataset Summary
st.subheader("Dataset Summary")
st.write(df.describe())

# Visualization 1: Population by District
if "district" in df.columns and "population" in df.columns:
    st.subheader("Population by District")
    district_population = df.groupby('district')['population'].sum().reset_index()
    district_population = district_population.sort_values(by='population', ascending=False)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(district_population['district'], district_population['population'], color='skyblue')
    plt.xlabel('District')
    plt.ylabel('Population')
    plt.title('Population by District in Negeri Sembilan')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    st.pyplot(plt)
else:
    st.warning("Columns 'district' and 'population' are required for population analysis.")

# Visualization 2: Price Analysis (if price data exists)
if "price" in df.columns and "district" in df.columns and "item_code" in df.columns:
    st.subheader("Average Price Analysis by District and Item Code")
    avg_price_data = df.groupby(['district', 'item_code']).agg(
        avg_price=('price', 'mean')
    ).reset_index()

    # Display summary table
    summary_table = avg_price_data.pivot(index='district', columns='item_code', values='avg_price')
    st.write("Summary Table: Average Price by District and Item Code")
    st.dataframe(summary_table)

    # Visualization
    plt.figure(figsize=(12, 6))
    sns.barplot(data=avg_price_data, x='district', y='avg_price', hue='item_code', palette='viridis')
    plt.title('Average Price by District and Item Code')
    plt.xlabel('District')
    plt.ylabel('Average Price')
    plt.xticks(rotation=45)
    plt.legend(title='Item Code', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    st.pyplot(plt)
else:
    st.warning("Columns 'price', 'district', and 'item_code' are required for price analysis.")
