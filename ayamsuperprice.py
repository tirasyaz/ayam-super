import streamlit as st
import pandas as pd

import streamlit as st
import pandas as pd

# Path to your dataset in Google Drive
file_path = '/content/drive/My Drive/AyamSuper'

# Load the dataset
data = pd.read_csv(file_path)

# Title and description
st.title("AyamSuper Streamlit App")
st.write("This is a visualization of AyamSuper project.")

# Add some interactive components (e.g., sliders, charts)
st.write("Data Overview")
st.dataframe(data.head())

# Add other visualizations as required (e.g., charts, maps)


from google.colab import drive
drive.mount('/content/drive')
'/content/drive/MyDrive/AyamSuper'
import os
import pandas as pd

folder_path = '/content/drive/MyDrive/AyamSuper'  # Update this with your folder path

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):  # Check if the file is a CSV
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        print(f"Data from {filename}:")
        print(df.head())  # Print the first few rows of each file

dataframes = []  # List to store each DataFrame

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)  # Append each DataFrame to the list

import pandas as pd
import os

# Load and filter `lookup_premise.csv` for Negeri Sembilan state
lookup_premise_df = pd.read_csv('/content/drive/MyDrive/AyamSuper/lookup_premise.csv')
negeri_sembilan_premises = lookup_premise_df[lookup_premise_df['state'] == 'Negeri Sembilan']
negeri_sembilan_premise_codes = negeri_sembilan_premises['premise_code'].dropna().unique()
print("Filtered Negeri Sembilan Premises:")
print(negeri_sembilan_premises.head(), "\n")

# Load and filter `lookup_item.csv` for item_code 2 (AYAM BERSIH - SUPER)
lookup_item_df = pd.read_csv('/content/drive/MyDrive/AyamSuper/lookup_item.csv')
ayam_super_items = lookup_item_df[lookup_item_df['item_code'] == 2]
print("Filtered AYAM BERSIH - SUPER Items:")
print(ayam_super_items.head(), "\n")

# Initialize an empty list to store filtered data
filtered_data = []

# Define the path to the folder containing the pricecatcher files
pricecatcher_folder_path = '/content/drive/MyDrive/AyamSuper'

# Filter each pricecatcher file for item_code 2 and matching premise_code
for filename in os.listdir(pricecatcher_folder_path):
    if filename.startswith("pricecatcher_") and filename.endswith(".csv"):
        pricecatcher_df = pd.read_csv(os.path.join(pricecatcher_folder_path, filename))
        
        # Filter for item_code 2 and premise_code in Negeri Sembilan
        filtered_df = pricecatcher_df[
            (pricecatcher_df['item_code'] == 2) &
            (pricecatcher_df['premise_code'].isin(negeri_sembilan_premise_codes))
        ]
        
        # Print the filtered data from the current file
        print(f"Filtered data from {filename}:")
        print(filtered_df.head(), "\n")
        
        # Append the filtered data
        filtered_data.append(filtered_df)

# Concatenate all filtered data from each file into a single DataFrame
final_filtered_data = pd.concat(filtered_data, ignore_index=True)

# Display the first few rows of the final filtered data
print("Final Concatenated Filtered Data:")
print(final_filtered_data.head())

import matplotlib.pyplot as plt
import seaborn as sns

# Ensure dates are in datetime format for plotting
final_filtered_data['date'] = pd.to_datetime(final_filtered_data['date'])

# Set up the plot style
sns.set(style="whitegrid")

# Plot the price trend for each premise_code in Negeri Sembilan
plt.figure(figsize=(14, 8))
sns.lineplot(data=final_filtered_data, x='date', y='price', hue='premise_code', marker='o', palette="tab10")

# Customize the plot
plt.title("Price Trend of 'AYAM BERSIH - SUPER' in Negeri Sembilan Premises")
plt.xlabel("Date")
plt.ylabel("Price (RM)")
plt.legend(title="Premise Code", loc='upper right')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Extract the month and year for grouping
final_filtered_data['year_month'] = final_filtered_data['date'].dt.to_period('M')

# Calculate the average price per month
monthly_avg_price = final_filtered_data.groupby('year_month')['price'].mean().reset_index()
monthly_avg_price['year_month'] = monthly_avg_price['year_month'].dt.to_timestamp()  # Convert back to datetime

# Set up the plot style
sns.set(style="whitegrid")

# Plot the average monthly price
plt.figure(figsize=(12, 6))
sns.barplot(data=monthly_avg_price, x='year_month', y='price', palette="Blues_d")

# Customize the plot
plt.title("Average Monthly Price of 'AYAM BERSIH - SUPER'")
plt.xlabel("Month")
plt.ylabel("Average Price (RM)")
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
