# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('Unemployment_in_India.csv')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Basic Information about the dataset
print("\nDataset Info:")
print(data.info())

# Descriptive statistics
print("\nDataset Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values in Dataset:")
print(data.isnull().sum())

# Strip leading and trailing spaces from column names
data.columns = data.columns.str.strip()

# Confirm the updated column names
print(data.columns)

# Convert the Date column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Distribution of Unemployment Rate
plt.figure(figsize=(10, 6))
sns.histplot(data['Estimated Unemployment Rate (%)'], kde=True, bins=30, color='blue')
plt.title('Distribution of Estimated Unemployment Rate (%)')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

import warnings
warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated")  # Ignore warning temporarily

plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='Date', y='Estimated Unemployment Rate (%)', hue='Region', marker='o')
plt.title('Unemployment Rate Trends by Region')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
correlation_matrix = data[['Estimated Unemployment Rate (%)','Estimated Employed','Estimated Labour Participation Rate (%)']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Grouped Analysis by Region
region_group = data.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values()
print("\nAverage Unemployment Rate by Region:")
print(region_group)

# Visualization of Regional Unemployment Rates
plt.figure(figsize=(12, 6))
region_group.plot(kind='bar', color='orange')
plt.title('Average Unemployment Rate by Region')
plt.xlabel('Region')
plt.ylabel('Average Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()