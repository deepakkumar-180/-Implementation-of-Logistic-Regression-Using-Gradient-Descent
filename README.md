# Implementation-of-Logistic-Regression-Using-Gradient-Descent
# NAME: DEEPAKKUMAR S
# REG NO: 212225230042
## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize the weights and bias with zero.

2.Calculate the sigmoid function and predicted probability.

3.Calculate the error between actual and predicted values.

4.Update the weights and bias using Gradient Descent until the specified number of iterations is completed.

## Program:

```

import pandas as pd
import numpy as np

# Load CSV file
data = pd.read_csv("placement.csv")

# Select input features and target
X = data[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']]
y = data['status'].map({'Placed': 1, 'Not Placed': 0})

# Convert to NumPy arrays
X = X.values
y = y.values

# Add bias column
X = np.c_[np.ones(X.shape[0]), X]

# Normalize features
X[:, 1:] = (X[:, 1:] - X[:, 1:].mean(axis=0)) / X[:, 1:].std(axis=0)

# Initialize weights
weights = np.zeros(X.shape[1])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient Descent
learning_rate = 0.01
iterations = 10000

for i in range(iterations):
    z = np.dot(X, weights)
    prediction = sigmoid(z)

    gradient = np.dot(X.T, (prediction - y)) / len(y)
    weights = weights - learning_rate * gradient

# Prediction
predicted = (sigmoid(np.dot(X, weights)) >= 0.5).astype(int)

# Accuracy
accuracy = np.mean(predicted == y)
print("Accuracy:", accuracy)

# Get student input
print("\nEnter student details:")

ssc = float(input("SSC Percentage: "))
hsc = float(input("HSC Percentage: "))
degree = float(input("Degree Percentage: "))
etest = float(input("E-Test Percentage: "))
mba = float(input("MBA Percentage: "))

# Normalize input using training data
input_data = np.array([ssc, hsc, degree, etest, mba])
mean = data[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']].mean().values
std = data[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']].std().values

input_data = (input_data - mean) / std

# Add bias
input_data = np.insert(input_data, 0, 1)

# Predict
probability = sigmoid(np.dot(input_data, weights))

print("\nPlacement Probability:", probability)

if probability >= 0.5:
    print("Prediction: Placed")
else:
    print("Prediction: Not Placed")


```

## Output:

<img width="999" height="757" alt="image" src="https://github.com/user-attachments/assets/7af5a75b-b071-47cc-9244-d80254064cc4" />




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

