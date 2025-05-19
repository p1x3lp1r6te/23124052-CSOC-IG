# Multivariable Linear Regression Project
# Author: Yogesh Kumar | Roll No: 23124052

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

# ----------------------
# Data Preprocessing
# ----------------------
data = pd.read_csv("housing.csv")
data = data.dropna(subset=['total_bedrooms'])
data = pd.get_dummies(data, columns=['ocean_proximity'])
ocean_cols = [col for col in data.columns if col.startswith("ocean_proximity_")]
data[ocean_cols] = data[ocean_cols].astype(float)

# Drop low-correlation features
drop_columns = [
    'households', 'total_bedrooms', 'population',
    'longitude', 'latitude',
    'ocean_proximity_ISLAND', 'ocean_proximity_INLAND'
]
data.drop(columns=[col for col in drop_columns if col in data.columns], inplace=True)

# Correlation Matrix
correlation = data.corr(numeric_only=True)
print("Correlation with median_house_value:\n")
print(correlation['median_house_value'].sort_values(ascending=False))

# ----------------------
# Splitting and Normalizing
# ----------------------
X = data.drop('median_house_value', axis=1)
y = data['median_house_value']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_mean = X_train.mean()
X_std = X_train.std()
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std
y_mean = y_train.mean()
y_std = y_train.std()
y_train_norm = (y_train - y_mean) / y_std

# ----------------------
# Part 1: Pure Python Linear Regression
# ----------------------
def gradient_descent_py(weights_now, b_now, X, y, L):
    n_samples = len(X)
    n_features = len(X.columns)
    weights_gradient = [0.0] * n_features
    b_gradient = 0.0
    for i in range(n_samples):
        row = list(X.iloc[i])
        prediction = sum(w * x for w, x in zip(weights_now, row)) + b_now
        error = prediction - y[i]
        for j in range(n_features):
            weights_gradient[j] += (1 / n_samples) * row[j] * error
        b_gradient += (1 / n_samples) * error
    new_weights = [w - L * gw for w, gw in zip(weights_now, weights_gradient)]
    new_b = b_now - L * b_gradient
    return new_weights, new_b

def predict_py(X, weights, bias):
    return [sum(w * x for w, x in zip(weights, row)) + bias for row in X.values]

weights_py = [0.0] * X_train.shape[1]
b_py = 0.0
L_py = 0.01
epochs = 500
train_mse_py, val_mse_py = [], []
start_py = time.time()
for _ in range(epochs):
    weights_py, b_py = gradient_descent_py(weights_py, b_py, X_train, y_train_norm.tolist(), L_py)
    train_pred_py = [(p * y_std + y_mean) for p in predict_py(X_train, weights_py, b_py)]
    val_pred_py = [(p * y_std + y_mean) for p in predict_py(X_val, weights_py, b_py)]
    train_mse_py.append(mean_squared_error(y_train, train_pred_py))
    val_mse_py.append(mean_squared_error(y_val, val_pred_py))
train_time_py = time.time() - start_py

# ----------------------
# Part 2: NumPy Implementation
# ----------------------
def gradient_descent_np(weights, b, X, y, L):
    predictions = X.dot(weights) + b
    errors = predictions - y
    weights -= L * X.T.dot(errors) / len(X)
    b -= L * errors.mean()
    return weights, b

def predict_np(X, weights, b):
    return X.dot(weights) + b

weights_np = np.zeros(X_train.shape[1])
b_np = 0.0
L_np = 0.005
train_mse_np, val_mse_np = [], []
start_np = time.time()
for _ in range(epochs):
    weights_np, b_np = gradient_descent_np(weights_np, b_np, X_train.values, y_train_norm.values, L_np)
    train_pred_np = predict_np(X_train.values, weights_np, b_np) * y_std + y_mean
    val_pred_np = predict_np(X_val.values, weights_np, b_np) * y_std + y_mean
    train_mse_np.append(mean_squared_error(y_train, train_pred_np))
    val_mse_np.append(mean_squared_error(y_val, val_pred_np))
train_time_np = time.time() - start_np

# ----------------------
# Part 3: Scikit-learn Implementation
# ----------------------
start_sk = time.time()
model = LinearRegression()
model.fit(X_train, y_train)
train_time_sk = time.time() - start_sk
train_pred_sk = model.predict(X_train)
val_pred_sk = model.predict(X_val)

# ----------------------
# Evaluation
# ----------------------
def evaluate(y_true, y_pred):
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }

print("\n--- Evaluation Metrics ---")
print(f"Pure Python Time: {train_time_py:.2f}s", evaluate(y_val, val_pred_py))
print(f"NumPy Time: {train_time_np:.2f}s", evaluate(y_val, val_pred_np))
print(f"Scikit-learn Time: {train_time_sk:.4f}s", evaluate(y_val, val_pred_sk))

# ----------------------
# Convergence Plots
# ----------------------
plt.figure(figsize=(10, 6))
plt.plot(train_mse_py, label='Train MSE (Pure Python)')
plt.plot(val_mse_py, label='Val MSE (Pure Python)')
plt.title('Convergence - Pure Python')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(train_mse_np, label='Train MSE (NumPy)')
plt.plot(val_mse_np, label='Val MSE (NumPy)')
plt.title('Convergence - NumPy')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

