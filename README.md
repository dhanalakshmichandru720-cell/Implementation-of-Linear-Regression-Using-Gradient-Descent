# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Generate Sample Data
# -----------------------------
np.random.seed(42)
X = 2 * np.random.rand(100, 1)   # Feature values
y = 4 + 3 * X + np.random.randn(100, 1)  # Target with noise

# -----------------------------
# 2. Gradient Descent Function
# -----------------------------
def gradient_descent(X, y, learning_rate=0.1, n_iterations=1000):
    m = len(y)  # number of samples
    X_b = np.c_[np.ones((m, 1)), X]  # add bias term (x0 = 1)
    theta = np.random.randn(2, 1)    # random initialization

    cost_history = []

    for iteration in range(n_iterations):
        gradients = (2/m) * X_b.T.dot(X_b.dot(theta) - y)
        theta -= learning_rate * gradients

        # Compute cost (MSE)
        cost = (1/m) * np.sum((X_b.dot(theta) - y) ** 2)
        cost_history.append(cost)

    return theta, cost_history

# -----------------------------
# 3. Train Model
# -----------------------------
theta_final, cost_history = gradient_descent(X, y, learning_rate=0.1, n_iterations=1000)

print("Final parameters (theta):", theta_final.ravel())
print("Final cost:", cost_history[-1])

# -----------------------------
# 4. Predictions
# -----------------------------
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_final)

# -----------------------------
# 5. Visualization
# -----------------------------
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="blue", label="Training data")
plt.plot(X_new, y_predict, color="red", linewidth=2, label="Prediction line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Simple Linear Regression using Gradient Descent")
plt.show()

# -----------------------------
# 6. Cost Convergence Plot
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(len(cost_history)), cost_history, color="green")
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Cost Function Convergence")
plt.show()
.
Developed by: dhanalakshmi.c
RegisterNumber:  25018616
*/
```

## Output:
<img width="1439" height="684" alt="Screenshot (147)" src="https://github.com/user-attachments/assets/0bc20a8e-f5f1-4305-96af-70b16df55dc3" />

<img width="1920" height="1080" alt="Screenshot (147)" src="https://github.com/user-attachments/assets/2a00736e-df55-4fdc-8b80-7ae8c804be24" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
