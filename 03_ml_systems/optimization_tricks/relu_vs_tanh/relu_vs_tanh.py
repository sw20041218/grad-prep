import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ========== 1. 数据 ==========
N = 200
X = np.linspace(-3, 3, N).reshape(-1, 1)
y = np.sin(X)

# ========== 2. 激活函数 ==========
def tanh(x):
    return np.tanh(x)

def tanh_grad(a):
    return 1 - a**2

def relu(x):
    return np.maximum(0, x)

def relu_grad(z):
    return (z > 0).astype(float)

# ========== 3. 训练函数 ==========
def train(activation="tanh", steps=2000, lr=0.01):
    H = 16
    W1 = 0.1 * np.random.randn(H, 1)
    b1 = np.zeros(H)
    W2 = 0.1 * np.random.randn(1, H)
    b2 = np.zeros(1)

    losses = []

    for _ in range(steps):
        z1 = X @ W1.T + b1

        if activation == "tanh":
            a1 = tanh(z1)
        else:
            a1 = relu(z1)

        y_pred = a1 @ W2.T + b2
        loss = np.mean((y_pred - y)**2)
        losses.append(loss)

        # backward
        dy = 2 * (y_pred - y) / N
        dW2 = dy.T @ a1
        db2 = dy.sum(axis=0)

        da1 = dy @ W2

        if activation == "tanh":
            dz1 = da1 * tanh_grad(a1)
        else:
            dz1 = da1 * relu_grad(z1)

        dW1 = dz1.T @ X
        db1 = dz1.sum(axis=0)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    return losses

# ========== 4. 运行对比 ==========
loss_tanh = train("tanh")
loss_relu = train("relu")

# ========== 5. 可视化 ==========
plt.figure(figsize=(8, 5))
plt.plot(loss_tanh, label="tanh", linewidth=2)
plt.plot(loss_relu, label="ReLU", linewidth=2)
plt.yscale("log")
plt.xlabel("Training Step")
plt.ylabel("MSE Loss (log scale)")
plt.title("tanh vs ReLU Optimization")
plt.legend()
plt.grid(True)
plt.show()