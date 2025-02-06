import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np

st.title("Interactive Data Analysis App")

# Sidebar for file upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.warning("Please upload a CSV file to begin analysis.")
    st.stop()

# Sidebar for variable selection
st.sidebar.header("Select Variables for Analysis")
all_columns = df.columns.tolist()
selected_variables = st.sidebar.multiselect("Choose one or more variables", all_columns, default=all_columns[:1])

# Main Section: Dataset Overview
st.subheader("Dataset Overview")
st.write(df[selected_variables].head())

# Descriptive Statistics
st.subheader("Descriptive Statistics")
st.write(df[selected_variables].describe())

# Advanced Descriptive Statistics
st.subheader("Advanced Descriptive Statistics")
if st.checkbox("Show Skewness and Kurtosis"):
    skewness = df[selected_variables].skew()
    kurtosis = df[selected_variables].kurt()
    st.write("Skewness:")
    st.write(skewness)
    st.write("Kurtosis:")
    st.write(kurtosis)

# Data Cleaning: Handling Missing Data
if st.sidebar.checkbox('Handle Missing Data'):
    fill_method = st.sidebar.radio('Select filling method', ('Mean', 'Median', 'Remove'))
    if fill_method == 'Mean':
        df[selected_variables] = df[selected_variables].fillna(df[selected_variables].mean())
    elif fill_method == 'Median':
        df[selected_variables] = df[selected_variables].fillna(df[selected_variables].median())
    else:
        df = df.dropna(subset=selected_variables)

# Histogram of Selected Variables
st.subheader("Histogram of Selected Variables")
fig, ax = plt.subplots(figsize=(10, 5))
for var in selected_variables:
    sns.histplot(df[var], kde=True, bins=50, ax=ax, label=var, alpha=0.6)
ax.legend()
st.pyplot(fig)

# PCA Scatter Plot
st.subheader("PCA Scatter Plot")
if len(selected_variables) > 1:
    pca = PCA(n_components=2)
    scaled_data = StandardScaler().fit_transform(df[selected_variables].dropna())
    pca_result = pca.fit_transform(scaled_data)
    fig, ax = plt.subplots()
    ax.scatter(pca_result[:, 0], pca_result[:, 1])
    ax.set_title("PCA Visualization")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)
else:
    st.warning("PCA requires at least 2 variables. Please select more variables.")

# Correlation Heatmap
st.subheader("Correlation Heatmap of Selected Variables")
if len(selected_variables) > 1:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[selected_variables].corr(), cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)
else:
    st.warning("Correlation heatmap requires at least 2 variables. Please select more variables.")

# Population vs. Sample Distribution
st.subheader("Population vs. Sample Distribution")
max_sample_size = min(len(df), 100)
sample_size = st.sidebar.slider("Sample Size", min_value=1, max_value=max_sample_size, value=min(10, max_sample_size))
sample = df[selected_variables].sample(n=sample_size, replace=False)
fig, ax = plt.subplots()
for var in selected_variables:
    sns.histplot(df[var], kde=True, color='blue', label=f'Population - {var}', alpha=0.6)
    sns.histplot(sample[var], kde=True, color='red', label=f'Sample - {var}', alpha=0.6)
ax.legend()
st.pyplot(fig)

# Hypothesis Testing (e.g., t-test)
st.subheader("Hypothesis Testing")
group1 = st.sidebar.selectbox("Select Group 1 (for t-test)", df[selected_variables].dropna().columns)
group2 = st.sidebar.selectbox("Select Group 2 (for t-test)", df[selected_variables].dropna().columns)

if st.sidebar.button("Perform t-test"):
    group1_data = df[group1].dropna()
    group2_data = df[group2].dropna()
    t_stat, p_val = stats.ttest_ind(group1_data, group2_data)
    st.write(f"T-Statistic: {t_stat}")
    st.write(f"P-Value: {p_val}")
    if p_val < 0.05:
        st.write("Reject null hypothesis (significant difference).")
    else:
        st.write("Fail to reject null hypothesis (no significant difference).")

# Download filtered data
st.sidebar.download_button("Download Filtered Data", df[selected_variables].to_csv(), file_name="filtered_data.csv", mime="text/csv")

# File download of visuals (Optional)
if st.sidebar.button("Download PCA Plot"):
    fig.savefig('pca_plot.png')
    with open("pca_plot.png", "rb") as file:
        st.sidebar.download_button(label="Download PCA Plot", data=file, file_name="pca_plot.png", mime="image/png")

st.write("Explore your dataset interactively!")