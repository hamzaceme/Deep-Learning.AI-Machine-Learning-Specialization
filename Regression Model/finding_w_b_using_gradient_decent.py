import numpy as np

# 1. Dataset
x_train = np.array([1.0, 2.0]) # House size in 1000 sqft[cite: 1]
y_train = np.array([300.0, 500.0]) # Price in $1000s[cite: 1]

# 2. Initialize parameters
w = 0.0
b = 0.0

# Hyperparameters
alpha = 0.1       # Learning rate
iterations = 1000 # Number of training steps

m = len(x_train)  # Number of training examples[cite: 1]

# 3. Training Loop (Gradient Descent)
for step in range(iterations):
    f_wb = w * x_train + b
    
    dj_dw = (1 / m) * np.sum((f_wb - y_train) * x_train)
    dj_db = (1 / m) * np.sum(f_wb - y_train)
    
    w = w - alpha * dj_dw
    b = b - alpha * dj_db

print(f"Trained parameters: w = {w:.2f}, b = {b:.2f}\n")

# ----------------------------------------------------
# 4. PREDICTING FOR A RANDOM / CUSTOM INPUT
# ----------------------------------------------------

# Define a random input value (e.g., 1.5 for a 1500 sqft house)
x_random = 1.5 

# Calculate prediction using f(x) = w * x + b
y_predicted = w * x_random + b

print(f"For an input size of {x_random} (1,500 sqft):")
print(f"Predicted Price = ${y_predicted:.2f} thousand dollars (${y_predicted * 1000:,.0f})")
