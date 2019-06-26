#!/usr/bin/env python
# coding: utf-8
# For this assessment we will need the following libraries and modules
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
from sklearn.metrics import mean_squared_error, r2_score


# Import data
raw_data = pd.read_csv('real_estate.csv')
raw_data = raw_data.drop(['No'],axis=1)  # Drop column 1 - index


# Categorize all transaction date in 2012 as 0 and 2013 as 1, so transaction date will not influence the data much
X1_one_hot_encoding = [1 if values >= 2013 else 0 for values in raw_data['X1 transaction date']]
raw_data['X1 transaction date'] = X1_one_hot_encoding


# Dealing with outliers by removing 0.5%, or 1% of the problematic samples.
condition = raw_data['X3 distance to the nearest MRT station'].quantile(0.99)
raw_data_no_outliers= raw_data[raw_data['X3 distance to the nearest MRT station']<condition]


# Reset index for observations removed
data_cleaned = raw_data_no_outliers.reset_index(drop=True)


# Let's transform X & Y with a log transformation
log_price = np.log(data_cleaned['Y house price of unit area'])
log_distance = np.log(data_cleaned['X3 distance to the nearest MRT station'])
data_cleaned['X3 log distance to the nearest MRT station'] = log_distance
data_cleaned['Y log house price of unit area'] = log_price
data_cleaned = data_cleaned.drop(['Y house price of unit area'],axis=1)
data_cleaned = data_cleaned.drop(['X3 distance to the nearest MRT station'],axis=1)


# Prepare data for ML and create test set
np.random.seed(42)
targets = data_cleaned['Y log house price of unit area']
inputs = data_cleaned.drop(['Y log house price of unit area'],axis=1)


# Data scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)


# Train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)


# Select and train a model
selection=-1
while selection not in ["1","2"]:
    selection=input('Which model do you want to use? [1] Linear Regression, [2] Random Forest \n')
    if selection not in ["1","2"]:
       print("Please only input 1 or 2")

if selection == "1":
   # Training with Linear regression model
   from sklearn.linear_model import LinearRegression
   reg = LinearRegression()
   reg.fit(x_train, y_train)
   
   # Check the outputs of the regression
   y_hat_lr = reg.predict(x_train)
   
   # Compute R2
   train_r2_lr=r2_score(y_train, y_hat_lr)
   
   # Compute RMSE
   train_mse_lr = mean_squared_error(y_train, y_hat_lr)
   train_rmse_lr = np.sqrt(train_mse_lr)
   print ("On training set, the RMSE using linear regression is",train_rmse_lr)
   print ("Please wait while modelling is being applied to test set....")

   # Test with linear regressor model
   reg.fit(x_test, y_test)
   y_hat_test_lr = reg.predict(x_test)
   
   # Compute R2
   test_r2_lr=r2_score(y_test, y_hat_test_lr)
   
   # Compute RMSE
   test_mse_lr = mean_squared_error(y_test, y_hat_test_lr)
   test_rmse_lr = np.sqrt(test_mse_lr)
   print ("On test set, the RMSE using linear regression is",test_rmse_lr)


elif selection == "2":
   # Training with random forest regressor model
   from sklearn.ensemble import RandomForestRegressor
   forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
   forest_reg.fit(x_train, y_train)
   
   # Compute RMSE
   y_hat_rf = forest_reg.predict(x_train)
   train_mse_rf = mean_squared_error(y_train, y_hat_rf)
   train_rmse_rf = np.sqrt(train_mse_rf)
   
   # Compute R2
   train_r2_rf=r2_score(y_train, y_hat_rf)
   
   # Perform CV with 10-fold
   def display_scores(scores):
       print("Scores:", scores)
       print("Mean:", scores.mean())
       print("Standard deviation:", scores.std())
   from sklearn.model_selection import cross_val_score
   scores_rf = cross_val_score(forest_reg, x_train, y_train, scoring="neg_mean_squared_error", cv=10)
   rmse_scores_rf = np.sqrt(-scores_rf)
   display_scores(rmse_scores_rf)
   print ("On training set, the RMSE using random forest with 10 fold is",rmse_scores_rf.mean())
   print ("Please wait while modelling is being applied to test set....")

   # Test with random forest regressor
   
   # Grid Search (fine-tune)
   # Search for best combination of hyperparameter values for the RandomForestRegressor:
   from sklearn.model_selection import GridSearchCV
   param_grid = [{'n_estimators': [10, 100, 200], 'max_features': [2, 3, 6], 'max_depth':[9,11,13]}]
   forest_reg = RandomForestRegressor(random_state=42)
   grid_search = GridSearchCV(forest_reg, param_grid, cv=10,scoring='neg_mean_squared_error', return_train_score=True, n_jobs=-1, verbose=2)
   grid_search.fit(x_train, y_train)
   grid_search.best_params_
   grid_search.best_estimator_
   
   # Score of each hyperparameter combination tested during the grid search
   cvres = grid_search.cv_results_
   print ("Score of each hyperparameter combination tested during the grid search")
   for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
       print(np.sqrt(-mean_score), params)
   
   # Apply RF model to test set
   final_model = grid_search.best_estimator_
   y_hat_test_rf = final_model.predict(x_test)
   
   # Compute final R2
   test_r2_rf=r2_score(y_test, y_hat_test_rf)
   
   # Compute final RMSE
   test_mse_rf = mean_squared_error(y_test, y_hat_test_rf)
   test_rmse_rf = np.sqrt(test_mse_rf)
   print("The RMSE using random forest with 10-fold is",test_rmse_rf, "The best hyperparameter is ", grid_search.best_params_)
