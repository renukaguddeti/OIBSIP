# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
data = pd.read_csv('car_data.csv')

# Data preprocessing
data['Car_Age'] = 2024 - data['Year']  # Calculate car age
data = data.drop(['Car_Name', 'Year'], axis=1)  # Drop unnecessary columns

# One-hot encoding for categorical features
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Splitting data into features (X) and target (y)
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Feature importance
coefficients = pd.DataFrame(model.coef_, index=X.columns, columns=['Coefficient'])
print("\nFeature Coefficients:")
print(coefficients)

# Visualization 1: Actual vs Predicted Selling Price
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--', linewidth=2)
plt.title('Actual vs Predicted Selling Price')
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.grid()
plt.show()

# Visualization 2: Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
