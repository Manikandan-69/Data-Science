import numpy as np

# Initialize the weights
w0 = 0.3
w1 = 0.73
#w2 = 1

# Learning rate
alpha = 0.1

# Data points
x1 = np.array([1,2,3])
#x2 = np.array([22, 24, 15, 20, 16])
y = np.array([1,3,5])

# Number of data points
m = len(y)

# Hypothesis function
def hypothesis(w0, w1, x1):
    return w0 + w1 * x1 

# Mean Squared Error function
def mse(w0, w1, x1, y):
    predictions = hypothesis(w0, w1, x1)
    return np.mean((predictions - y) ** 2 / 2)

# Gradient descent update function
def gradient_descent(w0, w1, x1, y, alpha, iterations):
    m = len(y)
    #re=[]
    for _ in range(iterations):
        predictions = hypothesis(w0, w1, x1 )
        error = predictions - y
        w0 -= alpha * np.sum(error) / m
        w1 -= alpha * np.sum(error * x1) / m
        #w2 -= alpha * np.sum(error * x2) / m
        #re.append([w0, w1, w2])
    return w0, w1
    #return re

# Initial mean squared error
initial_mse = mse(w0, w1, x1, y)
print(f"Initial MSE: {initial_mse}")

# Perform 2 iterations of gradient descent
w0, w1= gradient_descent(w0, w1,x1, y, alpha, 1)


# Final mean squared error after 2 iterations
#final_mse = mse(w0, w1, w2, x1 y)
#print(f"Final MSE after 2 iterations: {final_mse}")

# Final weights after 2 iterations
print(f"Weights after 2 iterations: w0 = {w0}, w1 = {w1}")
