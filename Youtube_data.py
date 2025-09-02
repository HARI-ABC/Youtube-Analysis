#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("C:\\Users\\haris\\Downloads\\youtube_channel_real_performance_analytics.csv")

# Preview the dataset
print(data.head())

# Display basic information about the dataset
print(data.info())

# Check for null values
print(data.isnull().sum())

# Fill or drop null values
data = data.dropna()


import isodate
def convert_duration(x):
    if isinstance(x, str):  # Only parse if it's a string
        return isodate.parse_duration(x).total_seconds()
    return float(x)  # Already a number

data['Video Duration'] = data['Video Duration'].apply(convert_duration)


import seaborn as sns
import matplotlib.pyplot as plt
# Exploratory Data Analysis (EDA)
sns.pairplot(data[['Views', 'Subscribers', 'Estimated Revenue (USD)']])
plt.show()
# Correlation Heatmap

plt.figure(figsize=(8,4))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.show()

#Top Performers by Revenue:

top_videos = data.sort_values(by='Estimated Revenue (USD)',
ascending=False).head(10)
print(top_videos[['ID', 'Estimated Revenue (USD)', 'Views',
'Subscribers']])


# Feature Engineering
data['Revenue per View'] = data['Estimated Revenue (USD)'] / data['Views']
data['Engagement Rate'] = (data['Likes'] + data['Shares'] + data['Comments']) / data['Views'] * 100


#data visualization
sns.histplot(data['Estimated Revenue (USD)'], bins=50, kde=True, color='green')
plt.show()

#Views vs Revenue:
sns.scatterplot(x=data['Views'], y=data['Estimated Revenue (USD)'])
plt.show()

#Build a Predictive Model
# Select features and target
features = ['Views', 'Subscribers', 'Likes', 'Shares',
'Comments', 'Engagement Rate']
target = 'Estimated Revenue (USD)'
X = data[features]
y = data[target]
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=42)

#Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100,
random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

#Feature Importance
importances = model.feature_importances_
sns.barplot(x=importances, y=features)
plt.show()

#Save Your Model
import joblib
joblib.dump(model, 'youtube_revenue_predictor.pkl')



