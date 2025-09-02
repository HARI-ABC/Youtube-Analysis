import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv("C:\\Users\\haris\\Downloads\\youtube_channel_real_performance_analytics (1).csv") 

# Preview the dataset
print(data.head())
# Display basic information about the dataset
print(data.info())
# Check for null values
print(data.isnull().sum())

# Convert 'Video Publish Time' to datetime format
data['Video Publish Time'] = pd.to_datetime(data['Video Publish Time'])

import matplotlib.pyplot as plt
import seaborn as sns


# Distribution of video durations
plt.figure(figsize=(10, 6))
sns.histplot(data=['Video Duration'], bins=30, kde=False, color='blue')
plt.title('Distribution of Video Durations')
plt.xlabel('Video Duration (seconds)')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(df['Estimated Revenue (USD)'], bins=30, kde=True)
plt.title('Distribution of Estimated Revenue')
plt.xlabel('Estimated Revenue (USD)')
plt.ylabel('Frequency')
plt.show()

# Select only numeric columns
numeric_data = data.select_dtypes(include=[np.number])

# Compute the correlation matrix
corr = numeric_data.corr()

# Plot the heatmap
plt.figure(figsize=(8,8))
sns.heatmap(corr, cmap='coolwarm', annot=False, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X = numeric_data.drop(columns=['Estimated Revenue (USD)'])
y = numeric_data['Estimated Revenue (USD)']

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the prediction accuracy
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmse


