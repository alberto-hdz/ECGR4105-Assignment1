import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset with your specific path
data = pd.read_csv(r"C:\Users\ahern\OneDrive\Documents\Programming\ECGR4105\assignment1\D3.csv")
X1 = data.iloc[:, 0].values  # First column
X2 = data.iloc[:, 1].values  # Second column
X3 = data.iloc[:, 2].values  # Third column
Y = data.iloc[:, 3].values   # Fourth column

# Normalize features for better convergence
def normalize_features(X):
    return (X - np.mean(X)) / np.std(X)

X1 = normalize_features(X1)
X2 = normalize_features(X2)
X3 = normalize_features(X3)
Y = normalize_features(Y)

# Gradient Descent Function for Problem 1
def gradient_descent(X, Y, theta, learning_rate, iterations):
    m = len(Y)
    loss_history = []
    
    for _ in range(iterations):
        prediction = X * theta
        gradient = (1/m) * np.sum(X * (prediction - Y))
        theta = theta - learning_rate * gradient
        loss = (1/(2*m)) * np.sum((prediction - Y) ** 2)
        loss_history.append(loss)
    
    return theta, loss_history

# Problem 1: Individual regressions
learning_rates = [0.1, 0.05, 0.01]
iterations = 1000

# Store results
models = {}
losses = {}

for lr in learning_rates:
    # X1 regression
    theta1, loss1 = gradient_descent(X1, Y, 0, lr, iterations)
    models[f'x1_lr{lr}'] = theta1
    losses[f'x1_lr{lr}'] = loss1
    
    # X2 regression
    theta2, loss2 = gradient_descent(X2, Y, 0, lr, iterations)
    models[f'x2_lr{lr}'] = theta2
    losses[f'x2_lr{lr}'] = loss2
    
    # X3 regression
    theta3, loss3 = gradient_descent(X3, Y, 0, lr, iterations)
    models[f'x3_lr{lr}'] = theta3
    losses[f'x3_lr{lr}'] = loss3

# Plotting for Problem 1
for lr in learning_rates:
    # X1
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.scatter(X1, Y, color='blue', alpha=0.5)
    plt.plot(X1, X1 * models[f'x1_lr{lr}'], color='red')
    plt.title(f'X1 Regression (lr={lr})')
    plt.xlabel('X1')
    plt.ylabel('Y')
    
    plt.subplot(132)
    plt.plot(losses[f'x1_lr{lr}'])
    plt.title(f'X1 Loss (lr={lr})')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    
    # X2
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.scatter(X2, Y, color='blue', alpha=0.5)
    plt.plot(X2, X2 * models[f'x2_lr{lr}'], color='red')
    plt.title(f'X2 Regression (lr={lr})')
    plt.xlabel('X2')
    plt.ylabel('Y')
    
    plt.subplot(132)
    plt.plot(losses[f'x2_lr{lr}'])
    plt.title(f'X2 Loss (lr={lr})')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    
    # X3
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.scatter(X3, Y, color='blue', alpha=0.5)
    plt.plot(X3, X3 * models[f'x3_lr{lr}'], color='red')
    plt.title(f'X3 Regression (lr={lr})')
    plt.xlabel('X3')
    plt.ylabel('Y')
    
    plt.subplot(132)
    plt.plot(losses[f'x3_lr{lr}'])
    plt.title(f'X3 Loss (lr={lr})')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.show()

# Problem 2: Multiple regression
def gradient_descent_multi(X, Y, theta, learning_rate, iterations):
    m = len(Y)
    loss_history = []
    X = np.column_stack((X, np.ones(m)))  # Add bias term
    
    for _ in range(iterations):
        prediction = X.dot(theta)
        gradient = (1/m) * X.T.dot(prediction - Y)
        theta = theta - learning_rate * gradient
        loss = (1/(2*m)) * np.sum((prediction - Y) ** 2)
        loss_history.append(loss)
    
    return theta, loss_history

X_multi = np.column_stack((X1, X2, X3))
best_theta = None
best_loss = float('inf')
best_lr = None
loss_history_multi = {}

for lr in learning_rates:
    theta = np.zeros(4)  # 3 variables + bias
    theta, loss = gradient_descent_multi(X_multi, Y, theta, lr, iterations)
    loss_history_multi[lr] = loss
    if loss[-1] < best_loss:
        best_loss = loss[-1]
        best_theta = theta
        best_lr = lr

# Plot loss for Problem 2
plt.figure()
for lr in learning_rates:
    plt.plot(loss_history_multi[lr], label=f'lr={lr}')
plt.title('Loss for Multiple Regression')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predictions for new values
new_X = np.array([[1, 1, 1], [2, 0, 4], [3, 2, 1]])
new_X = (new_X - np.mean([X1, X2, X3], axis=1)) / np.std([X1, X2, X3], axis=1)  # Normalize
new_X = np.column_stack((new_X, np.ones(3)))
predictions = new_X.dot(best_theta)

# Print results
print("Problem 1 Results:")
for lr in learning_rates:
    print(f"Learning Rate {lr}:")
    print(f"X1 model: y = {models[f'x1_lr{lr}']:.4f}x1")
    print(f"X2 model: y = {models[f'x2_lr{lr}']:.4f}x2")
    print(f"X3 model: y = {models[f'x3_lr{lr}']:.4f}x3")
    print(f"Final losses: X1={losses[f'x1_lr{lr}'][-1]:.4f}, X2={losses[f'x2_lr{lr}'][-1]:.4f}, X3={losses[f'x3_lr{lr}'][-1]:.4f}")
    print()

print("Problem 2 Results:")
print(f"Best model (lr={best_lr}): y = {best_theta[0]:.4f}x1 + {best_theta[1]:.4f}x2 + {best_theta[2]:.4f}x3 + {best_theta[3]:.4f}")
print(f"Predictions: {predictions}")