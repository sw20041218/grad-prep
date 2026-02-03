import numpy as np

# Define two vectors in R^3
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Vector addition
v_sum = v1 + v2

# Scalar multiplication
v_scaled = 2 * v1

print("v1:", v1)
print("v2:", v2)
print("v1 + v2:", v_sum)
print("2 * v1:", v_scaled)

# Linear mapping represented by a matrix
A = np.array([
    [1, 0, 0],
    [0, 2, 0],
    [0, 0, 3]
])

mapped_v1 = A @ v1
print("A @ v1:", mapped_v1)
