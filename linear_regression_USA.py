# Linear Regression Project on USA Housing Dataset

# Importing required libraries
import numpy as np                          # For numerical operations
import pandas as pd                         # For handling dataset
import matplotlib.pyplot as plt             # For visualization
import seaborn as sns                       # For advanced visualization

# Sklearn imports
from sklearn.model_selection import train_test_split  # For train-test split
from sklearn.linear_model import LinearRegression     # For Linear Regression model
from sklearn import metrics                           # For model evaluation


# Load the Dataset
df = pd.read_csv("USA_Housing.csv")   # Load the dataset from CSV file

# Show first 5 rows of the dataset
print(df.head())

# Get complete info about dataset (column names, null values, datatypes)
print(df.info())

# Get statistical summary (mean, std, min, max etc.)
print(df.describe())

# Show all column names
print(df.columns)


# Data Visualization

# Pairplot - shows pairwise relationships between columns
sns.pairplot(df)
plt.show()

# Distribution plot of 'Price' column (target variable)
sns.histplot(df['Price'], kde=True)
plt.show()

# Heatmap for correlation between features
sns.heatmap(df.drop("Address", axis=1).corr(), annot=True, cmap="coolwarm")
plt.show()


# Preparing Data for Machine Learning

# Select independent variables (features) - remove 'Price' and 'Address'
X = df.drop(['Price', 'Address'], axis=1)

# Dependent variable (label)
y = df['Price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=101
)
# test_size=0.4 means 40% data for testing, 60% for training
# random_state=101 ensures reproducibility of results

# Training the Linear Regression Model

# Initialize model
lm = LinearRegression()

# Fit the model on training data
lm.fit(X_train, y_train)

# Print intercept value
print("Intercept:", lm.intercept_)

# Print coefficients for each feature
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)


# Predictions and Evaluation

# Make predictions on the test dataset
predictions = l(X_tem.predictst)

# Print sample predictions
print("Sample Predictions:", predictions[:10])

# Scatter plot of actual vs predicted values
plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# Residuals distribution (difference between actual and predicted)
sns.histplot((y_test - predictions), kde=True)
plt.title("Residuals Distribution")
plt.show()


# Error Evaluation

# Mean Absolute Error (MAE)
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, predictions))

# Mean Squared Error (MSE)
print("Mean Squared Error:", metrics.mean_squared_error(y_test, predictions))

# Root Mean Squared Error (RMSE)
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, predictions)))
