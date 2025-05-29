# BUILD-A-STOCK-MARKET-ANALYSER
#importing dataset
#new start
import yfinance as yf

# List of NIFTY50 tickers for 2023
nifty50_tickers = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS",
    "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "HCLTECH.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS",
    "TATAMOTORS.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "TECHM.NS", 
    "INDUSINDBK.NS", "JSWSTEEL.NS", "ADANIENT.NS", "COALINDIA.NS", "NESTLEIND.NS",
    "BAJAJFINSV.NS", "GRASIM.NS", "TATASTEEL.NS", "M&M.NS", "BPCL.NS",
    "DRREDDY.NS", "BRITANNIA.NS", "CIPLA.NS", "DIVISLAB.NS", "HINDALCO.NS",
    "HEROMOTOCO.NS", "APOLLOHOSP.NS", "EICHERMOT.NS", "UPL.NS", "ADANIPORTS.NS",
    "SBILIFE.NS", "BAJAJ-AUTO.NS", "TATACONSUM.NS", "ICICIGI.NS"
]

# Download Nifty50 data for the year 2023
nifty50_data = yf.download(nifty50_tickers, start="2023-01-01", end="2023-12-31")

# Print first few rows of data
print(nifty50_data.head())

# Optionally, save the data to a CSV file
csv_path = r"C:\Users\poorna chandu\Downloads\nifty50_data_2023.csv"
nifty50_data.to_csv(csv_path)

print(f"Data for Nifty50 tickers saved to: {csv_path}")

# Extract only 'Close' prices and drop rows with NaNs
close_prices = nifty50_data['Close'].dropna()

# Transpose: Each row = date, each column = company
close_prices = close_prices.T
print(close_prices.head())

#Normalization
#different types of normalization techniques
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer

# Load your CSV file
csv_path = r"C:\Users\poorna chandu\Downloads\nifty50_data_2023.csv"
nifty50_data = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)

# Clean the data (remove any rows with NaNs in the 'Close' price for any stock)
cleaned_data = nifty50_data.dropna(subset=[('Close', ticker) for ticker in nifty50_data.columns.levels[1]])

# Extract the 'Close' prices
close_prices = cleaned_data['Close']

# Transpose the data (companies as rows, dates as columns)
close_prices = close_prices.T
# ✅ Define scaled_dfs here
scaled_dfs = {}

#1 Z-score Normalization
z_scaler = StandardScaler()
z_scaled = z_scaler.fit_transform(close_prices)
z_df = pd.DataFrame(z_scaled, columns=close_prices.columns, index=close_prices.index)
print("\nZ-Score Scaled Data:\n", z_df.head())

# Save Z-Score Scaled Data to CSV
z_df.to_csv(r"C:\Users\poorna chandu\Downloads\Z_scaled_nifty50.csv")

# 2 Min-Max Scaling (scales values between 0 and 1)
minmax_scaler = MinMaxScaler()
minmax_scaled = minmax_scaler.fit_transform(close_prices)
minmax_df = pd.DataFrame(minmax_scaled, columns=close_prices.columns, index=close_prices.index)
print("\nMin-Max Scaled Data:\n", minmax_df.head())

# Save Min-Max Scaled Data to CSV
minmax_df.to_csv(r"C:\Users\poorna chandu\Downloads\minmax_scaled_nifty50.csv")

# 3 Max-Abs Scaling (scales by maximum absolute value)
maxabs_scaler = MaxAbsScaler()
maxabs_scaled = maxabs_scaler.fit_transform(close_prices)
maxabs_df = pd.DataFrame(maxabs_scaled, columns=close_prices.columns, index=close_prices.index)
print("\nMax-Abs Scaled Data:\n", maxabs_df.head())

# Save Max-Abs Scaled Data to CSV
maxabs_df.to_csv(r"C:\Users\poorna chandu\Downloads\maxabs_scaled_nifty50.csv")

# 4 Robust Scaling (uses median and IQR)
robust_scaler = RobustScaler()
robust_scaled = robust_scaler.fit_transform(close_prices)
robust_df = pd.DataFrame(robust_scaled, columns=close_prices.columns, index=close_prices.index)
print("\nRobust Scaled Data:\n", robust_df.head())

# Save Robust Scaled Data to CSV
robust_df.to_csv(r"C:\Users\poorna chandu\Downloads\robust_scaled_nifty50.csv")

# 5 L2 Normalization (scales each row to unit norm)
l2_normalizer = Normalizer(norm='l2')
l2_normalized = l2_normalizer.fit_transform(close_prices)
l2norm_df = pd.DataFrame(l2_normalized, columns=close_prices.columns, index=close_prices.index)
print("\nL2 Normalized Data:\n", l2norm_df.head())

# Save L2 Normalized Data to CSV
l2norm_df.to_csv(r"C:\Users\poorna chandu\Downloads\l2norm_scaled_nifty50.csv")

print("\n✅ All normalization techniques applied and CSVs saved successfully.")

import matplotlib.pyplot as plt

# Example stocks to plot (make sure these are present in your data)
example_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']  # You can change these

# All normalized DataFrames
scaled_dfs = {
    'Z-Score': z_df,
    'Min-Max': minmax_df,
    'Max-Abs': maxabs_df,
    'Robust': robust_df,
    'L2-Norm': l2norm_df
}

# Plotting
for method_name, df in scaled_dfs.items():
    plt.figure(figsize=(14, 6))

    # Plot each stock's normalized values across time
    for stock in example_stocks:
        if stock in df.index:
            stock_values = df.loc[stock].to_numpy()  # ✅ convert to 1D array
            dates = df.columns.to_numpy()            # ✅ ensure dates are 1D too
            plt.plot(dates, stock_values, label=stock, linewidth=2)

    plt.title(f"{method_name} Normalization - Nifty 50 Close Prices", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #finding best normalization technique
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer


# Load your data (assuming 'close_prices' is already defined as per your earlier code)
csv_path = r"C:\Users\poorna chandu\Downloads\nifty50_data_2023.csv"
nifty50_data = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)


scalers = {
    'Z-Score': StandardScaler(),
    'Min-Max': MinMaxScaler(),
    'Max-Abs': MaxAbsScaler(),
    'Robust': RobustScaler(),
    'L2 Norm': Normalizer(norm='l2')
}

k_range = range(2, 100)
results = {}

for name, scaler in scalers.items():
    scaled = pd.DataFrame(scaler.fit_transform(close_prices),
                          index=close_prices.index, columns=close_prices.columns)

    silhouette_scores = []
    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scaled)
            score = silhouette_score(scaled, labels)
            silhouette_scores.append(score)
        except Exception as e:
            silhouette_scores.append(np.nan)

    results[name] = silhouette_scores

# Create DataFrame from results
results_df = pd.DataFrame(results, index=[f'k={k}' for k in k_range])

# Determine best method per k (row-wise)
results_df['Best Method per k'] = results_df.idxmax(axis=1)

# Compute average scores (excluding the string column)
average_scores = results_df.drop(columns='Best Method per k').mean()
best_overall = average_scores.idxmax()

print("Average Silhouette Scores Across k:")
print(average_scores)
print(f"\n✅ Best Overall Normalization Method: {best_overall}")


plt.figure(figsize=(8, 5))
average_scores.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title("Average Silhouette Score per Normalization Method")
plt.ylabel("Average Silhouette Score")
plt.xlabel("Normalization Method")
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# 🧯 Optional: Fix for MKL/OpenBLAS threading error on Windows
os.environ["OMP_NUM_THREADS"] = "1"

# ---- STEP 1: Prepare the Data ----
# Assuming 'nifty50_data' is already loaded and contains a multi-index with ['Close'] prices
close_prices = nifty50_data['Close'].dropna()
close_prices = close_prices.T

# ---- STEP 2: Normalize the Data ----
#1 Z-score Normalization
z_scaler = StandardScaler()
z_scaled = z_scaler.fit_transform(close_prices)
z_df = pd.DataFrame(z_scaled, columns=close_prices.columns, index=close_prices.index)
# 2 Min-Max Scaling (scales values between 0 and 1)
minmax_scaler = MinMaxScaler()
minmax_scaled = minmax_scaler.fit_transform(close_prices)
minmax_df = pd.DataFrame(minmax_scaled, columns=close_prices.columns, index=close_prices.index)

# 3 Max-Abs Scaling (scales by maximum absolute value)
maxabs_scaler = MaxAbsScaler()
maxabs_scaled = maxabs_scaler.fit_transform(close_prices)
maxabs_df = pd.DataFrame(maxabs_scaled, columns=close_prices.columns, index=close_prices.index)

# 4 Robust Scaling (uses median and IQR)
robust_scaler = RobustScaler()
robust_scaled = robust_scaler.fit_transform(close_prices)
robust_df = pd.DataFrame(robust_scaled, columns=close_prices.columns, index=close_prices.index)

# 5 L2 Normalization (scales each row to unit norm)
l2_normalizer = Normalizer(norm='l2')
l2_normalized = l2_normalizer.fit_transform(close_prices)
l2norm_df = pd.DataFrame(l2_normalized, columns=close_prices.columns, index=close_prices.index)


# ---- STEP 3: Silhouette Score for Different k ----
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

silhouette_scores = []
k_range = range(2, 11)  # Silhouette score not defined for k=1

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(robust_df)
    score = silhouette_score(robust_df, cluster_labels)
    silhouette_scores.append(score)
    print(f"k = {k}, silhouette score = {score:.4f}")

# ---- STEP 4: Plot Silhouette Scores ----
plt.figure(figsize=(8, 5))
plt.plot(k_range, silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Method to Determine Optimal k')
plt.grid(True)
plt.show()

#k-means Clustering
from sklearn.cluster import KMeans

# Choose number of clusters
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(robust_df)

# Add cluster labels to the DataFrame
robust_df['Cluster'] = clusters

# Display stocks by cluster
for cluster_id in range(2):
    stocks_in_cluster = robust_df[robust_df['Cluster'] == cluster_id].index.tolist()
    print(f"\nCluster {cluster_id + 1} stocks:\n{stocks_in_cluster}")

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(robust_df.drop('Cluster', axis=1))  # exclude Cluster column

# Step 2: Create a new DataFrame with PCA results
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
pca_df['Stock'] = robust_df.index
pca_df['Cluster'] = robust_df['Cluster'].values  # Add cluster labels

# Step 3: Plot the PCA components with clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], 
                      c=pca_df['Cluster'], cmap='tab10', s=100)

# Optional: Add stock name labels
for i, row in pca_df.iterrows():
    plt.text(row['PC1'] + 0.1, row['PC2'], row['Stock'], fontsize=8)

plt.title("PCA Visualization of Stock Clusters (KMeans)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()

#3DPCA
from sklearn.decomposition import PCA
import plotly.express as px

# Step 1: Apply PCA with 3 components (excluding the 'Cluster' column)
pca_3d = PCA(n_components=3)
pca_components_3d = pca_3d.fit_transform(robust_df.drop('Cluster', axis=1))

# Step 2: Create a new DataFrame for PCA components and cluster labels
pca_df_3d = pd.DataFrame(data=pca_components_3d, columns=['PC1', 'PC2', 'PC3'])
pca_df_3d['Cluster'] = robust_df['Cluster'].values
pca_df_3d['Stock'] = robust_df.index  # Add stock names for hover info

# Step 3: Plot interactive 3D scatter plot
fig = px.scatter_3d(
    pca_df_3d,
    x='PC1',
    y='PC2',
    z='PC3',
    color='Cluster',
    hover_name='Stock',
    title='3D PCA Cluster Visualization',
    color_continuous_scale='Viridis',
    opacity=0.8
)

fig.update_traces(marker=dict(size=6))
fig.show()


