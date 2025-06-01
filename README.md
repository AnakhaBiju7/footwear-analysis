# footwear-analysis
Analyze footwear e-commerce data with EDA, predictive modeling (XGBoost), clustering (K-Means), anomaly detection, fairness analysis, and association rule mining using Python. Utilizes Pandas, Scikit-learn, SHAP, and MLxtend to uncover pricing trends, product insights, and market patterns for retailers and analysts.

The dataset, sourced from Kaggle, includes information on footwear products such as brand, color, size, original price, and offer price. The project is structured into the following key components:

- **Exploratory Data Analysis (EDA)**: Visualizes the distribution of original and offer prices, examines relationships between prices, and identifies the most popular brands through histograms, scatter plots, and count plots.
- **Preprocessing**: Cleans the dataset by handling missing values, converting categorical variables (brand, color) into numeric encodings using LabelEncoder, and normalizing numerical features (price, offer price, discount percentage, size) with StandardScaler.
- **Predictive Modeling**: Implements an XGBoost Regressor to predict offer prices based on features like brand, color, and size. Model performance is evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score. SHAP values are used to interpret feature importance.
- **Clustering**: Applies K-Means clustering to segment products into groups based on scaled features (price, offer price, discount, size). The optimal number of clusters is determined using the Elbow method, and results are visualized in a 2D PCA scatter plot.
- **Anomaly Detection**: Uses Isolation Forest to identify outliers in the dataset, such as unusually priced products. Anomalies are visualized using PCA for dimensionality reduction.
- **Fairness Analysis**: Assesses pricing fairness across brands, sizes, and colors by calculating a fairness index based on offer prices. Identifies the most overpriced and underpriced products and visualizes average fairness per brand.
- **Association Rule Mining**: Employs the Apriori algorithm to discover patterns between product attributes (brand, color, size) and discount levels, revealing insights into customer purchasing behavior.


## Requirements
To run this project, install the required Python packages:
pip install pandas numpy seaborn matplotlib scikit-learn xgboost shap mlxtend


## Dataset
This project uses a dataset named `data.csv`, which is sourced from Kaggle. The dataset is not included in this repository due to its size and Kaggle's usage terms. It contains the following columns:
- `brand`: Product brand (categorical)
- `color`: Product color (categorical)
- `size`: Product size (can be numeric or categorical)
- `price`: Original price (numeric)
- `offer_price`: Discounted price (numeric)

To run the code, you’ll need to download the dataset from Kaggle:
- **Kaggle Link**: (https://www.kaggle.com/datasets/ashutosh598/shoes-price-for-various-brands)
- Download the dataset from Kaggle 
