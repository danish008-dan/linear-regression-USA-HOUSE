USA Housing Price Prediction using Linear Regression
📌 Project Overview

This project demonstrates the implementation of a Linear Regression Model to predict house prices in the USA using the USA Housing dataset. The dataset contains information such as average income, average house age, average number of rooms, average number of bedrooms, population, and address.

The goal of this project is to build a machine learning model that can predict housing prices based on these features and evaluate its performance using standard regression metrics.

📂 Dataset

The dataset used is USA_Housing.csv, which contains the following columns:

Avg. Area Income – Average income of residents of the city

Avg. Area House Age – Average age of houses in the city

Avg. Area Number of Rooms – Average number of rooms per house

Avg. Area Number of Bedrooms – Average number of bedrooms per house

Area Population – Population of the area

Price – Price of the house (Target variable)

Address – Address of the house (not used in training)

🛠️ Technologies Used

Python

NumPy & Pandas – Data handling and preprocessing

Matplotlib & Seaborn – Data visualization

Scikit-Learn – Machine Learning model (Linear Regression) and evaluation

📊 Exploratory Data Analysis (EDA)

The project includes visualizations to better understand the dataset:

Pairplot showing pairwise relationships between variables

Distribution plot of housing prices

Correlation heatmap to identify relationships between features

⚙️ Model Training & Testing

The dataset is split into training (60%) and testing (40%) sets.

A Linear Regression Model is trained on the training dataset.

Model coefficients and intercept are displayed to interpret feature importance.

📈 Model Evaluation

The model is evaluated using:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

Additionally, scatter plots of Actual vs Predicted prices and residual distribution are shown for better performance analysis.

🚀 How to Run the Project

Clone the repository

git clone https://github.com/your-username/USA-Housing-LinearRegression.git
cd USA-Housing-LinearRegression


Install required libraries

pip install numpy pandas matplotlib seaborn scikit-learn


Run the Python script

python housing_regression.py

📌 Results

The model demonstrates a strong linear relationship between housing features and prices.

Predictions closely align with actual values with a reasonable error margin.

📜 Conclusion

This project highlights how Linear Regression can be effectively applied to real-world datasets for predictive analysis. The methodology used here can be extended to other regression problems in various domains.

🔮 Future Improvements

Apply other regression models (Ridge, Lasso, Polynomial Regression) for comparison.

Perform feature engineering to improve accuracy.

Deploy the model as a web application using Flask or Django.
