import numpy as np

np.random.seed(0)

# ========== 1) 造数据：y = sin(x) + noise ==========
N = 200
X = np.linspace(-3, 3, N).reshape(-1, 1)          # (N, 1)
y_true = np.sin(X) + 0.05 * np.random.randn(N, 1) # (N, 1)

# ========== 2) 初始化网络参数 ==========
H = 16  # 隐藏层宽度（神经元数）

W1 = 0.5 * np.random.randn(H, 1)  # (H, 1)
b1 = np.zeros((H,))               # (H,)
W2 = 0.5 * np.random.randn(1, H)  # (1, H)
b2 = np.zeros((1,))               # (1,)

lr = 0.05
steps = 3000

def tanh(x):
    return np.tanh(x)

def tanh_deriv(a):
    # a = tanh(z)，则 d/dz tanh(z) = 1 - tanh(z)^2 = 1 - a^2
    return 1.0 - a**2

# ========== 3) 训练：forward + backward + update ==========
for step in range(steps):
    # ---- forward ----
    z1 = X @ W1.T + b1          # (N, H)
    a1 = tanh(z1)               # (N, H)
    y_pred = a1 @ W2.T + b2     # (N, 1)

    # MSE loss
    diff = y_pred - y_true
    loss = np.mean(diff**2)

    # ---- backward ----
    # dLoss/dy_pred = 2/N * (y_pred - y_true)
    dY = (2.0 / N) * diff                     # (N, 1)

    # y_pred = a1 @ W2.T + b2
    dW2 = dY.T @ a1                           # (1, H)
    db2 = np.sum(dY, axis=0)                  # (1,)

    # dA1 = dY @ W2
    dA1 = dY @ W2                             # (N, H)

    # a1 = tanh(z1)
    dZ1 = dA1 * tanh_deriv(a1)                # (N, H)

    # z1 = X @ W1.T + b1
    dW1 = dZ1.T @ X                           # (H, 1)
    db1 = np.sum(dZ1, axis=0)                 # (H,)

    # ---- update ----
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if step % 300 == 0:
        print(f"step {step:4d} | loss {loss:.6f}")

print("\nTraining finished.")
print("Final loss:", loss)