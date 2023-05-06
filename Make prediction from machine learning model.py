# Created by: Muhammad Jihadil (jihadil003@gmail.com) 
# This code was write in Python 3.9
# The purpose of this code is to make pradiction using a machine learning model

# Importing the libraries 
import pandas as pd #Version 1.3.0
import joblib #Version 1.0.1

# Load the machine learning model
regressor = joblib.load("./model24978.joblib")

# Read data for testing
data_test = pd.read_csv('uji.csv')

# Test Data
x = data_test.iloc[:, 1:5] # 1st until 4th columns
y = data_test.iloc[:, 5] # 5th columns

# Make prediction
prediction = regressor.predict(x)
df = pd.DataFrame({"Ra": prediction})

# Save dataframe to csv
df.to_csv('prediction.csv', index=False)
