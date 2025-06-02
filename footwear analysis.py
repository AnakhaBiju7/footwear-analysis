# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load dataset (assuming data is available locally)
df = pd.read_csv('data (5).csv')

# --- Exploratory Data Analysis (EDA) and Visualization ---

print("--- EDA ---")
print("\nDataset Info:")
df.info()

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Description:")
print(df.describe())

print("\nMissing values before handling:")
print(df.isnull().sum())

print("\nUnique values in key categorical columns:")
print("Brands:", df['brand'].nunique())
print("Colors:", df['color'].nunique())
print("Sizes:", df['size'].nunique())

# Basic Visualizations
plt.figure(figsize=(12, 6))
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Distribution of Original Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(df['offer_price'], bins=50, kde=True)
plt.title('Distribution of Offer Price')
plt.xlabel('Offer Price')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(y='brand', data=df, order=df['brand'].value_counts().index[:10], hue=df['brand'], palette='viridis', legend=False)
plt.title('Top 10 Brands by Count')
plt.xlabel('Count')
plt.ylabel('Brand')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='offer_price', data=df, alpha=0.6)
plt.title('Original Price vs Offer Price')
plt.xlabel('Original Price')
plt.ylabel('Offer Price')
plt.grid(True)
plt.show()

# --- Preprocessing ---
print("\n--- Preprocessing ---")
# Convert 'size' to numeric, forcing errors to NaN
df['size_numeric'] = pd.to_numeric(df['size'], errors='coerce')
print(f"Rows with non-numeric size before dropping: {df['size_numeric'].isnull().sum()}")
df.dropna(subset=['size_numeric'], inplace=True)
print(f"Remaining rows after dropping non-numeric size: {len(df)}")

# Calculate discount_percent before scaling
df['discount_percent'] = ((df['price'] - df['offer_price']) / df['price'] * 100).round(2)
# Replace inf values if any arise from division by zero (price=0)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with missing values (including potential NaNs from discount calculation)
print(f"Missing values before final dropna:\n{df.isnull().sum()}")
initial_rows = len(df)
df.dropna(inplace=True)
print(f"Rows dropped due to missing values: {initial_rows - len(df)}")
print(f"Remaining rows after final dropna: {len(df)}")

# Label encode categorical columns
le_brand = LabelEncoder()
le_color = LabelEncoder()
df['brand_encoded'] = le_brand.fit_transform(df['brand'])
df['color_encoded'] = le_color.fit_transform(df['color'])

# Normalize continuous columns (use size_numeric instead of size)
scaler = StandardScaler()
df[['price_scaled', 'offer_price_scaled', 'discount_scaled', 'size_scaled']] = scaler.fit_transform(
    df[['price', 'offer_price', 'discount_percent', 'size_numeric']]
)

print("\nPreprocessing Complete. Data snapshot after preprocessing:")
print(df.head())
print("\nDataset Info after preprocessing:")
df.info()

# --- Predictive Modeling ---
print("\n--- Predictive Modeling ---")
# Features exclude 'discount_percent' to avoid data leakage
features = ['brand_encoded', 'color_encoded', 'size_numeric']
X = df[features]
y = df['offer_price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Regressor
model = XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("🔍 Test Set Performance:")
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)
print("\n🔍 Training Set Performance:")
print(f"MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}, R²: {train_r2:.4f}")

# SHAP values for feature importance
print("\n--- SHAP Feature Importance ---")
explainer = shap.Explainer(model)
shap_values = explainer(X)
if X.shape[0] > 1000:
    print("Plotting SHAP beeswarm for a sample of 1000 instances...")
    shap.plots.beeswarm(shap_values[:1000], show=False)
else:
    shap.plots.beeswarm(shap_values, show=False)
plt.title('SHAP Beeswarm Plot')
plt.show()

# --- Clustering ---
print("\n--- Clustering ---")
# Select features for clustering
cluster_data = df[['price_scaled', 'offer_price_scaled', 'discount_scaled', 'size_scaled']]

# Elbow method for optimal k
print("Running Elbow method for KMeans...")
wcss = []
max_k = min(10, len(cluster_data) - 1) if len(cluster_data) > 1 else 1
if max_k > 1:
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(cluster_data)
        wcss.append(kmeans.inertia_)

    # Plot elbow
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), wcss, marker='o')
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.show()
else:
    print("Not enough data points to perform Elbow method for k > 1.")

# Choose k=4 (assumed optimal based on typical elbow plot outcome or domain knowledge)
optimal_k = min(4, len(cluster_data))
if optimal_k >= 1:
    print(f"Applying KMeans with k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(cluster_data)

    # Reduce to 2D for visualization (only if enough components)
    if cluster_data.shape[1] >= 2:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(cluster_data)
        df['pca1'] = pca_result[:, 0]
        df['pca2'] = pca_result[:, 1]

        # Plot clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=df, palette='Set2', s=60)
        plt.title('Product Segmentation via K-Means')
        plt.grid(True)
        plt.show()
    else:
        print("Not enough features for 2D PCA visualization.")

    # Cluster summary
    print("\nCluster Summary:")
    if 'cluster' in df.columns:
        print(df.groupby('cluster')[['price', 'offer_price', 'discount_percent']].mean().round(2))
    else:
        print("Clustering was not performed due to insufficient data.")
else:
    print("Clustering requires at least one data point.")

# --- Anomaly Detection ---
print("\n--- Anomaly Detection ---")
anomaly_features = df[['price_scaled', 'offer_price_scaled', 'discount_scaled', 'size_scaled']]
if len(anomaly_features) > 1:
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    df['anomaly_score'] = iso_forest.fit_predict(anomaly_features)
    df['is_anomaly'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

    # Count and display anomalies
    num_anomalies = df['is_anomaly'].sum()
    print("\nNumber of anomalies detected:", num_anomalies)
    if num_anomalies > 0:
        anomalies = df[df['is_anomaly'] == 1]
        print("\nAnomalies:")
        print(anomalies[['brand', 'color', 'size', 'price', 'offer_price', 'discount_percent']])
    else:
        print("No anomalies detected.")

    # Visualize anomalies (only if pca results are available)
    if 'pca1' in df.columns and 'pca2' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='pca1', y='pca2', hue='is_anomaly', data=df, palette={0: 'blue', 1: 'red'}, s=60, alpha=0.6)
        plt.title('Anomaly Detection via Isolation Forest')
        plt.grid(True)
        plt.legend(title='Anomaly')
        plt.show()
    else:
        print("PCA results not available for anomaly visualization.")
else:
    print("Not enough data points to perform Anomaly Detection.")

# --- Fairness Analysis ---
print("\n--- Fairness Analysis ---")
if not df.empty:
    # Use offer_price for fairness index to align with simulation
    df['fairness_index'] = df.groupby(['brand', 'size', 'color'])['offer_price'].transform(
        lambda x: (x - x.median()) / (x.std() if x.std() != 0 else np.nan)
    )
    df['fairness_index'] = df['fairness_index'].fillna(0)

    # Top 10 most overpriced and underpriced
    print("\n🚨 Most Overpriced Products:")
    print(df.sort_values('fairness_index', ascending=False).head(10)[['brand', 'color', 'size', 'price', 'offer_price', 'fairness_index']])
    print("\n💸 Most Underpriced Products:")
    print(df.sort_values('fairness_index').head(10)[['brand', 'color', 'size', 'price', 'offer_price', 'fairness_index']])

    # Average fairness per brand
    brand_fairness = df.groupby('brand')['fairness_index'].mean().sort_values()
    if not brand_fairness.empty:
        plt.figure(figsize=(10, 6))
        brand_fairness.plot(kind='barh', color='skyblue')
        plt.title('Average Offer Price Fairness Index by Brand')
        plt.xlabel('Fairness Index (0 = fair)')
        plt.grid(True)
        plt.axvline(0, color='red', linestyle='--')
        plt.show()
    else:
        print("No data to plot Average Fairness per Brand.")
else:
    print("Not enough data to perform Fairness Analysis.")

# --- Association Rule Mining ---
print("\n--- Association Rule Mining ---")
if not df.empty:
    df['discount_level'] = pd.cut(df['discount_percent'],
                                 bins=[0, 25, 50, 75, 100],
                                 labels=['Low', 'Medium', 'High', 'Very High'],
                                 right=False,
                                 include_lowest=True)
    df['discount_level'] = df['discount_level'].astype(str).replace('nan', 'Unknown')
    df['size_numeric_str'] = df['size_numeric'].astype(str)

    transactions = df[['brand', 'color', 'size_numeric_str', 'discount_level']].values.tolist()
    transactions = [[str(item) for item in transaction if item is not None and pd.notna(item)] for transaction in transactions]
    transactions = [t for t in transactions if t]

    if transactions:
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

        if not df_encoded.empty:
            try:
                frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
                if not frequent_itemsets.empty:
                    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
                    rules_sorted = rules.sort_values('lift', ascending=False)
                    print("\nTop 10 Association Rules:")
                    if not rules_sorted.empty:
                        print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
                    else:
                        print("No association rules found above the specified threshold.")
                else:
                    print("No frequent itemsets found with the specified min_support.")
            except Exception as e:
                print(f"An error occurred during Apriori or Association Rules generation: {e}")
        else:
            print("Encoded transaction DataFrame is empty. Cannot perform association rule mining.")
    else:
        print("No valid transactions found after preprocessing.")
else:
    print("Not enough data to perform Association Rule Mining.")

print("\n--- Analysis Complete ---")
