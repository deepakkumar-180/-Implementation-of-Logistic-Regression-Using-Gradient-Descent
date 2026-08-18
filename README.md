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

import numpy as np

# Input data
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])

# Initialize parameters
w = 0.0
b = 0.0

learning_rate = 0.1
iterations = 1000

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient Descent
for i in range(iterations):

    # Prediction
    z = w * X[:, 0] + b
    y_pred = sigmoid(z)

    # Calculate gradients
    dw = np.mean((y_pred - y) * X[:, 0])
    db = np.mean(y_pred - y)

    # Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

# Predict function
def predict(x):
    probability = sigmoid(w * x + b)
    return 1 if probability >= 0.5 else 0

# Display results
print("Weight:", w)
print("Bias:", b)

print("\nPredictions:")
for x in X[:, 0]:
    print("Input:", x, "Predicted Class:", predict(x))

# Predict new value
new_value = 3.5
print("\nPrediction for", new_value, ":", predict(new_value))

```

## Output:

<img width="947" height="559" alt="image" src="https://github.com/user-attachments/assets/fb9efa79-9e6a-42d5-92b7-222415ad92aa" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

