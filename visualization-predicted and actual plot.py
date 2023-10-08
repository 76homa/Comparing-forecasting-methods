import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models (use your trained models here)
models = {
    'Neural Networks': MLPRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Polynomial Regression degree 6': LinearRegression()
}

# Create scatter plots for each model's predictions vs. actual values
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Scatter Plot for {model_name}')
    plt.show()
