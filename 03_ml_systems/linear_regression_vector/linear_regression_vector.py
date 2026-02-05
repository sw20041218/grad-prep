import numpy as np

np.random.seed(0)

# ========== 1) 造数据：y = Xw + b + noise ==========
n = 200          # 样本数
d = 3            # 特征维度（向量形式的核心）

true_w = np.array([2.0, -1.0, 0.5])
true_b = 1.0

X = np.random.randn(n, d)  # (n, d)
noise = np.random.normal(0, 0.3, size=n)
y = X @ true_w + true_b + noise  # (n,)

# ========== 2) 初始化参数 ==========
w = np.zeros(d)  # (d,)
b = 0.0

lr = 0.05
steps = 800

# ========== 3) 梯度下降训练 ==========
for step in range(steps):
    y_pred = X @ w + b                  # (n,)
    error = y_pred - y                  # (n,)
    loss = np.mean(error ** 2)          # scalar

    # 对 w 的梯度：d/dw mean((Xw+b-y)^2) = 2/n * X^T (Xw+b-y)
    dw = (2.0 / n) * (X.T @ error)      # (d,)
    # 对 b 的梯度：d/db mean((... )^2) = 2/n * sum(Xw+b-y)
    db = (2.0 / n) * np.sum(error)      # scalar

    w -= lr * dw
    b -= lr * db

    if step % 100 == 0:
        print(f"step {step:4d} | loss {loss:.4f} | w {w} | b {b:.4f}")

print("\nTraining finished")
print("Estimated w:", w)
print("Estimated b:", b)
print("True w:", true_w)
print("True b:", true_b)