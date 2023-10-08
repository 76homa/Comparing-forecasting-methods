import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Load your dataset
# Replace this with your actual dataset loading code
excel_file_path = 'C:\\Users\\homa.behmardi\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Anaconda3 (64-bit)\\clustered_data\\cluster_0.xlsx'
sheet_name = 'Sheet1'
data = pd.read_excel(excel_file_path, sheet_name=sheet_name)

# Replace X and y with your feature and target variable
X = data[['DL_PRB_Utilization_Rate(Ericsson_LTE_Sector)', 'Average_Reported_CQI(Ericsson_LTE_Sector)', 'DL_Spectral_efficiency(Ericsson_LTE_Sector)']]
y = data['Average_UE_DL_Throughput(Mbps)(Ericsson_LTE_Sector)']

# Define models
models = {
    'Neural Networks': MLPRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Polynomial Regression degree 6': LinearRegression()
}

# Initialize K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and calculate MSE for each model
mse_scores = {}

for model_name, model in models.items():
    mse = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    mse_scores[model_name] = -mse.mean()

# Print the MSE scores for each model
for model_name, mse_score in mse_scores.items():
    print(f"{model_name}: Mean Squared Error (MSE) = {mse_score:.4f}")

# Choose the model with the lowest MSE
best_model = min(mse_scores, key=mse_scores.get)
print(f"The best model is: {best_model}")
