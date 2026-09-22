#Universal & Generalized Version (For Any Scenario)Below is a general, clean template utilizing vectorized NumPy operations (eliminating manual for loops) so you can plug in any custom $1\text{D}$ dataset ($x, y$) and parameters ($w, b$) with minimal adjustments:Pythonimport numpy as np

import matplotlib.pyplot as plt

def run_linear_regression_model(x_data, y_data, weight, bias, x_new=None, x_label="Feature (X)", y_label="Target (Y)"):
    """
    General Linear Regression prediction and visualization tool.
    
    Parameters:
    - x_data (list/ndarray): 1D array of feature values
    - y_data (list/ndarray): 1D array of target values
    - weight (float): Linear model weight (w)
    - bias (float): Linear model bias (b)
    - x_new (float/ndarray, optional): New input value(s) to make predictions for
    - x_label (str), y_label (str): Plot labels
    """
    # Convert inputs to NumPy arrays
    x = np.array(x_data, dtype=np.float64)
    y = np.array(y_data, dtype=np.float64)
    
    # Vectorized model output calculation: f(x) = w * x + b
    y_pred = weight * x + bias
    
    # Visualization
    plt.figure(figsize=(7, 4))
    plt.scatter(x, y, color='red', marker='x', label='Actual Data')
    plt.plot(x, y_pred, color='blue', label=f'Model Line: f(x) = {weight}x + {bias}')
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Linear Regression Model")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
    # Prediction for new input
    if x_new is not None:
        x_new_arr = np.array(x_new, dtype=np.float64)
        y_new_pred = weight * x_new_arr + bias
        print(f"Prediction for X = {x_new}: Y = {y_new_pred}")
        return y_new_pred

# ==========================================
# EXAMPLE USAGE (Modify these variables)
# ==========================================
X = [1.0, 2.0, 3.0, 4.0]               # Your custom inputs
Y = [300.0, 500.0, 700.0, 900.0]       # Your custom target labels
w_param = 200                          # Custom slope/weight
b_param = 100                          # Custom intercept/bias
new_input = 2.5                        # New x value to predict

run_linear_regression_model(
    x_data=X, 
    y_data=Y, 
    weight=w_param, 
    bias=b_param, 
    x_new=new_input,
    x_label="House Size (1000 sqft)",
    y_label="Price ($1000s)"
)
