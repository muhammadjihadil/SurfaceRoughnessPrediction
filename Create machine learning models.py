# Created by: Muhammad Jihadil (jihadil003@gmail.com) 
# This code was write in Python 3.9

# Publication link: https://digilib.itb.ac.id/index.php/gdl/view/61731/
# Full document of research (Indonesian): https://drive.google.com/file/d/11mNiniTwni6MafxhKD4aNdwxpY4vRQ_5/view?usp=share_link 
# Full document of research (English): https://drive.google.com/file/d/1Suzm-9u5XGUMgomDtiqgIgJm_-gwYU0L/view?usp=share_link

# The goal is RMSE_uji
# You can treat RMSE_latihan and delta_RMSE as qualytative parameters for certain combination of hyperparameters to  make judgement about overfittingness of a model

# Version 0.24.2 is used for Scikit-Learn library
# Importing the libraries
import pandas as pd #Version 1.3.0
import numpy as np #Version 1.21.1
import joblib #Version 1.0.1
import csv 
import os

# Read data
data_train = pd.read_csv('latihan.csv')
data_test = pd.read_csv('uji.csv')

# Training data
x1 = data_train.iloc[:, 1:5] # 1st until 4th columns
y1 = data_train.iloc[:, 5] # 5th columns

# Test data
x2 = data_test.iloc[:, 1:5] # 1st until 4th columns
y2 = data_test.iloc[:, 5] # 5th columns

# Hyperparameter search limit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
param_grid = {'min_samples_leaf': [int(x) for x in np.linspace(start = 1, stop = 20, num = 20)],
     'n_estimators': [int(x) for x in np.linspace(start = 2, stop = 625, num = 624)],
     'max_features': [1, 2, 3]}

# Make scorer (RSME)
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
def score_rmse(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared = False)
    return rmse
scorer = make_scorer(score_rmse, greater_is_better = False)

# Search hyperparameter
forest = RandomForestRegressor()
forest_gs = GridSearchCV(forest, param_grid, cv = 5,
                         scoring = scorer,
                         verbose = 3,
                         n_jobs = -1)
forest_gs.fit(x1, y1)

# Write result in output file
exportfile='hyperparameter.csv'
means = forest_gs.cv_results_['mean_test_score']
params = forest_gs.cv_results_['params']
with open(exportfile, 'w', newline='') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    header = ["max_features", "min_samples_leaf", "n_estimators", "RMSE_latihan"]
    writer.writerow(header)
    for mean, param in zip(means, params):
        writer.writerow([list(param.values()), mean])

# Cleaning file
text1 = open("hyperparameter.csv", "r")
text1 = ''.join([i for i in text1])
text1 = text1.replace('"[', "")
text1 = text1.replace(']"', "")
text1 = text1.replace(" ", "")
text1 = text1.replace("-", "")
text2 = open("hyperparameter.csv","w")
text2.writelines(text1)
text2.close()

# Make row counter
file = open("hyperparameter.csv", "r")
fileObject = csv.reader(file)
n = sum(1 for row in fileObject) # number of row

# Open hyperparameter file
hyperparameter = pd.read_csv('hyperparameter.csv')
hyperparameter.insert(4,'RMSE_uji','')
hyperparameter.insert(5,'delta_RMSE','')
hyperparameter.insert(6,'nomor_model','')

# Create machine learning models
for i in range(n-1):
    # Create variable for hyperparameter
    feature = hyperparameter.iloc[i, 0]
    leaf = hyperparameter.iloc[i, 1]
    estimator = hyperparameter.iloc[i, 2]

    # Create regressor object 
    regressor = RandomForestRegressor(min_samples_leaf = leaf,
                                      n_estimators = estimator, 
                                      max_features = feature,
                                      n_jobs=-1)
    
    # Fit the regressor with x and y data 
    regressor.fit(x1, y1)

    # Make test prediction
    prediction = regressor.predict(x2)

    # RMSE Uji
    rmse_uji = mean_squared_error(y2, prediction, squared = False)
    hyperparameter.iloc[i, 4] = rmse_uji

    # Delta RMSE
    hyperparameter.iloc[i, 5] = abs(rmse_uji - hyperparameter.iloc[i, 3])
    
    # Numbering the model
    hyperparameter.iloc[i, 6] = i+1

    # Save model
    directory = os.path.dirname(__file__)
    location = os.path.join(directory, 'data 25/model')
    filename = "\model" + str(i+1)
    print(filename)
    formatfile = ".joblib"
    savefile = location + filename + formatfile
    joblib.dump(regressor, savefile)

# Save dataframe to csv
hyperparameter.to_csv('hyperparameter.csv', index=False)
