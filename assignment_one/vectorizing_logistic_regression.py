import numpy as np
import time

# Set the random seed
np.random.seed(50)

# Example data for comparison (5 features, 100 examples)
X = np.random.rand(5, 100)  # 5 features, 100 examples
y = np.random.randint(0, 2, (1, 100))  # Random binary labels (0 or 1)

n_x, m = X.shape  # Number of features and number of examples
learning_rate = 0.01
iterations = 1000

### Non-vectorized logistic regression ###
def non_vectorized_logistic_regression(X, y, n_x, m, learning_rate, iterations):
    w = np.zeros((n_x, 1))  # Initialize weights
    b = 0  # Initialize bias
    
    print(f'{n_x}, {m}')

    # Train for 1000 iterations
    for iter in range(iterations):

        J = 0  # Cost
        dw = np.zeros((n_x, 1))  # Gradient for weights # Already in correct form (row vector)
        wt_xi = 0
        db = 0  # Gradient for bias

        # Traverse all 100 examples
        for i in range(m):
            # Forward propagation for each example
            # z_i = wT(x_i) + b

            for row in range(n_x):
                wt_xi += X[row, i]

            z_i = wt_xi + db # Linear function
            # a_i = sigma(z_i)
            a_i = 1 / (1 + np.exp(-z_i)) # Sigmoid function
            
            # Cost function for this example
            # (-[(y_i)log(a_i)+(1-y_i)log(1-a_i)])
            J += -(y[0, i] * np.log10(a_i) + (1 - y[0, i]) * np.log10(1 - a_i))

            # Backward propagation (gradients)
            dz_i = a_i - y[0, i]  # Scalar gradient

            # dw += X[i] * dz_i 
            for x_i in range(n_x): # Gradient for weights
                dw[x_i, 0] += X[x_i, i] * dz_i

            db += X # Gradient for bias

        # Average the cost and gradients
        J /= m # J 
        
        for x_i in range(n_x): # dw
            dw[x_i, 0] /= m

        db /= m # db 
        

        # Update weights and bias
        for i in range(n_x): # w
            w[i] -= (learning_rate * dw[i])
        
        b -= (learning_rate * db) # b

    return w, b, J

### Vectorized logistic regression ###
# def vectorized_logistic_regression(X, y, n_x, m, learning_rate, iterations):
#     w = np.zeros((n_x, 1))  # Initialize weights
#     b = 0  # Initialize bias

#     for iter in range(iterations):
#         # Forward propagation
#         Z = #Your Code Here  # Linear function (vectorized)
#         A = #Your Code Here  # Sigmoid function (vectorized)
        
#         # Cost function (vectorized)
#         J = #Your Code Here
        
#         # Backward propagation (vectorized)
#         dZ = #Your Code Here  # Gradient of cost with respect to Z
#         dw = #Your Code Here  # Gradient with respect to weights
#         db = #Your Code Here  # Gradient with respect to bias

#         # Update weights and bias
#         #Your Code Here # w 
#         #Your Code Here # b

#     return w, b, J

### Timing and execution ###

# Measure time for non-vectorized version
start_time = time.time()
w_non_vec, b_non_vec, J_non_vec = non_vectorized_logistic_regression(X, y, n_x, m, learning_rate, iterations)
non_vec_time = time.time() - start_time
print(f"Non-vectorized Logistic Regression Time: {non_vec_time:.6f} seconds")

# Measure time for vectorized version
start_time = time.time()
#  w_vec, b_vec, J_vec = vectorized_logistic_regression(X, y, n_x, m, learning_rate, iterations)
vec_time = time.time() - start_time
print(f"Vectorized Logistic Regression Time: {vec_time:.6f} seconds")

# Compare results
print(f"Cost from non-vectorized: {J_non_vec[0]:.6f}")
# print(f"Cost from vectorized: {J_vec:.6f}")