# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries and load the dataset from the CSV file.
2. Separate the independent variable (Level) and dependent variable (Salary).
3. Create and train the Decision Tree Regressor model using the training data.
4. Predict salary for new input and evaluate the model using R², MAE, MSE, and RMSE.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Jeensfer Jo
RegisterNumber:212225240058

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Step 1: Load the dataset
dataset = pd.read_csv("Salary.csv")

# Step 2: Separate features and target
X = dataset.iloc[:, 1:2].values   # Level
y = dataset.iloc[:, 2].values     # Salary

# Step 3: Create model
regressor = DecisionTreeRegressor(random_state=0)

# Step 4: Train model
regressor.fit(X, y)

# Step 5: Predict on training data (since dataset is very small)
y_pred = regressor.predict(X)

# Step 6: Calculate Metrics
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print("R2 Score :", r2)
print("MAE :", mae)
print("MSE :", mse)
print("RMSE :", rmse)

# Step 7: Predict for new value
level = 6.5
print("Predicted Salary for Level", level, "is:", regressor.predict([[level]])[0])
*/
```

## Output:
<img width="325" height="89" alt="image" src="https://github.com/user-attachments/assets/c718810b-f62a-4dc7-ae2d-a759a4fb66b9" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
