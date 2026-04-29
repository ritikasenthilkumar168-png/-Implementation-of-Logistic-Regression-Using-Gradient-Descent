# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Start with weights and bias as 0.
2. Calculate the output using the sigmoid function.
3. Find the error and adjust weights and bias.
4. Repeat this process until the result becomes accurate.
 
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv("Placement_Data.csv")

# Convert Placed / Not Placed to 1 / 0
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})

# Take only 2 features (simple)
X = data[['ssc_p', 'mba_p']].values
y = data['status'].values

# -----------------------------
# Standard Scaler (Normalization)
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add bias column (1)
m = len(y)
X = np.c_[np.ones(m), X]

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def cost_function(X, y, theta):
    h = sigmoid(X @ theta)
    return (-1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))

# Gradient Descent
theta = np.zeros(X.shape[1])
alpha = 0.1
cost_history = []

for i in range(500):
    z = X @ theta
    h = sigmoid(z)
    gradient = (1/m) * X.T @ (h - y)
    theta = theta - alpha * gradient
    
    cost = cost_function(X, y, theta)
    cost_history.append(cost)

# Prediction
y_pred = (sigmoid(X @ theta) >= 0.5).astype(int)

# Accuracy
accuracy = np.mean(y_pred == y) * 100
print("Wei

Developed by: Ritika S
RegisterNumber:  212225220086
*/
```
## Output:
![logistic regression using gradient descent](sam.png)
<img width="730" height="799" alt="Screenshot 2026-04-29 095040" src="https://github.com/user-attachments/assets/3c6a5e6a-91eb-4dc8-9865-1bcad8200947" />
## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

