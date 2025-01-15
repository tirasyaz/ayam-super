# Analysis on Chicken Price in Negeri Sembilan

This project aims to analyze chicken prices in Negeri Sembilan from October 2023 to October 2024 and make prediction on chicken price using LSTM models and visualize the results in an interactive Streamlit dashboard. 

---

## 1. Define the Goal
- **Objective**: To forecast chicken prices for three specific item codes and analyze price trends to assist stakeholders in making informed decisions.
- **Scope**: This project focuses on time-series price prediction for chicken using LSTM and visualizes the results through an interactive Streamlit application.

---

## 2. Data Acquisition
- **Source**: The dataset was obtained from [data.gov.my](https://www.data.gov.my).
- **Preprocessing**:
  - Removed missing values and irrelevant categorical variables.
  - Normalized the feature variables using Min-Max Scaling to bring all values into the [0, 1] range.
  - Separated the dataset into features (X) and the target variable (y).
  - Split the data into 80% training and 20% testing sets for model training and evaluation.

---

## 3. Descriptive Analysis
- **Summary Statistics**:
  - Calculated mean, median, mode, range, and standard deviation for key variables.
- **Key Observations**:
  - Identified seasonal trends, such as price peaks in certain months.
  - Detected anomalies, like a significant price increase.
    
---

## 4. Diagnostic Analysis
- **Investigation of Patterns**:
  - Analyzed correlations between variables to understand factors influencing price trends.
  - Used regression models to explore relationships between prices and time.

---

## 5. Predictive Analysis
- **Model Development**:
  - Built an LSTM model with two layers, each containing 20 units and a dropout rate of 20% to prevent overfitting.
  - Used a dense layer with 25 units and ReLU activation for feature extraction.
- **Evaluation Metrics**:
  - Accuracy: 80%
  - Precision: 79%
  - Recall: 80%
  - F1-Score: 79%
  - RMSE: Calculated to measure average prediction errors.
- **Predicted Trends**:
  - The predicted prices align closely with actual trends, with minor discrepancies in certain months.

---

## 6. Prescriptive Analysis
- **Recommendations**:
  - Monitor price fluctuations for Item 3, especially during peak periods like November.
  - Develop policies to stabilize prices during significant anomalies.
  - Use predicted trends to plan procurement and supply chain logistics effectively.

---

## 7. Visualization and Communication
- **Streamlit Dashboard**:
  - The results and visualizations are presented in an interactive Streamlit app.
  - Key features include:
    - Time-series plots comparing actual and predicted prices.
    - Statistical summaries and trend visualizations.
    - User-friendly interface for stakeholders of varying technical expertise.
  - To run the dashboard:
    ```bash
    streamlit run app.py
    ```
