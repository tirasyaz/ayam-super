import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from st_aggrid import AgGrid
import os

# Streamlit app title
st.title("Negeri Sembilan Population and Price Analysis")

# File paths for datasets (replace with your GitHub repository link if needed)
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

    # Display data table with Ag-Grid
    st.subheader("Population Data Table (Interactive)")
    AgGrid(district_population)

    # Interactive Population Distribution Plot (Plotly)
    st.subheader("Population Distribution (Interactive)")
    fig = px.bar(
        district_population, 
        x='district', 
        y='population', 
        title='Population by District in Negeri Sembilan', 
        labels={'district': 'District', 'population': 'Population'},
        color='population', 
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        xaxis_title='District',
        yaxis_title='Population',
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig)
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

    # Interactive Average price by district and item_code (Plotly)
    st.subheader("Average Price by District and Item Code")
    fig = px.bar(
        avg_price_data,
        x='district',
        y='avg_price',
        color='item_code',
        title='Average Price by District and Item Code',
        labels={'district': 'District', 'avg_price': 'Average Price'},
        barmode='group'
    )
    fig.update_layout(
        xaxis_title='District',
        yaxis_title='Average Price',
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig)
else:
    st.warning("Required columns for price analysis are not present in the dataset.")

# Add dynamic filters for the Population Distribution
st.sidebar.header("Filter Population Data")
selected_districts = st.sidebar.multiselect(
    "Select Districts to View",
    options=district_population['district'].unique(),
    default=district_population['district'].unique()
)

filtered_population = district_population[district_population['district'].isin(selected_districts)]

# Plot filtered data
st.subheader("Filtered Population Distribution")
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(filtered_population['district'], filtered_population['population'], color='skyblue')
ax.set_xlabel('District')
ax.set_ylabel('Population')
ax.set_title('Population by District (Filtered)')
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)

# Export option for filtered data
@st.cache
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(filtered_population)
st.download_button(
    label="Download Filtered Population Data",
    data=csv,
    file_name="filtered_population.csv",
    mime="text/csv"
)

# Add a slider to filter data based on price range (price data)
st.sidebar.header("Filter Price Data")
price_slider = st.sidebar.slider(
    "Select Price Range", 
    min_value=float(price_data['price'].min()),
    max_value=float(price_data['price'].max()),
    value=(float(price_data['price'].min()), float(price_data['price'].max()))
)

filtered_price_data = price_data[
    (price_data['price'] >= price_slider[0]) & 
    (price_data['price'] <= price_slider[1])
]

# Interactive Plot for filtered price data (Plotly)
st.subheader("Price Distribution (Filtered)")
fig = px.bar(
    filtered_price_data, 
    x='district', 
    y='price', 
    title='Price Distribution by District',
    labels={'district': 'District', 'price': 'Price'},
    color='district', 
    color_continuous_scale='Viridis'
)
fig.update_layout(
    xaxis_title='District',
    yaxis_title='Price',
    xaxis_tickangle=-45
)
st.plotly_chart(fig)

# Export option for filtered price data
csv_price = convert_df(filtered_price_data)
st.download_button(
    label="Download Filtered Price Data",
    data=csv_price,
    file_name="filtered_price_data.csv",
    mime="text/csv"
)

# Interactive summary table for price data with Ag-Grid
st.subheader("Interactive Price Data Table")
AgGrid(filtered_price_data)

# Summary and conclusion
st.markdown("""
    ### Summary
    - This app provides an interactive analysis of population and price data for districts in Negeri Sembilan.
    - Use the filters on the sidebar to customize the view.
    - You can download filtered data using the download buttons.
""")
