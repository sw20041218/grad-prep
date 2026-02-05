import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ========== 数据 ==========
N = 200
X = np.linspace(-3, 3, N).reshape(-1, 1)
y = np.sin(X)

# ========== 激活 ==========
def tanh(x):
    return np.tanh(x)

def tanh_grad(a):
    return 1 - a**2

def relu(x):
    return np.maximum(0, x)

def relu_grad(z):
    return (z > 0).astype(float)

# ========== 训练（返回 loss 序列） ==========
def train(activation="tanh", init_scale=0.1, steps=2000, lr=0.01, H=16):
    # 参数初始化：只由 init_scale 控制
    W1 = init_scale * np.random.randn(H, 1)
    b1 = np.zeros(H)
    W2 = init_scale * np.random.randn(1, H)
    b2 = np.zeros(1)

    losses = []
    for _ in range(steps):
        # forward
        z1 = X @ W1.T + b1
        if activation == "tanh":
            a1 = tanh(z1)
        else:
            a1 = relu(z1)

        y_pred = a1 @ W2.T + b2
        diff = y_pred - y
        loss = np.mean(diff**2)
        losses.append(loss)

        # backward
        dy = 2 * diff / N                  # (N,1)
        dW2 = dy.T @ a1                    # (1,H)
        db2 = dy.sum(axis=0)               # (1,)

        da1 = dy @ W2                      # (N,H)
        if activation == "tanh":
            dz1 = da1 * tanh_grad(a1)
        else:
            dz1 = da1 * relu_grad(z1)

        dW1 = dz1.T @ X                    # (H,1)
        db1 = dz1.sum(axis=0)              # (H,)

        # update
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

        # 简单防爆：如果炸了就提前停
        if not np.isfinite(loss):
            break

    return np.array(losses)

# ========== 多个 init_scale 对比 ==========
scales = [0.01, 0.1, 1.0, 3.0]
steps = 2000
lr = 0.01

fig, axes = plt.subplots(len(scales), 1, figsize=(9, 12), sharex=True)

for i, s in enumerate(scales):
    # 为公平起见，每个 scale 都重置随机种子，使对比更稳定
    np.random.seed(0)
    loss_t = train("tanh", init_scale=s, steps=steps, lr=lr)
    np.random.seed(0)
    loss_r = train("relu", init_scale=s, steps=steps, lr=lr)

    ax = axes[i]
    ax.plot(loss_t, label="tanh", linewidth=2)
    ax.plot(loss_r, label="ReLU", linewidth=2)
    ax.set_yscale("log")
    ax.set_ylabel(f"scale={s}\nloss")
    ax.grid(True)

axes[-1].set_xlabel("Training Step")
axes[0].set_title("Initialization Scale Sweep: tanh vs ReLU (log loss)")
axes[0].legend()
plt.tight_layout()
plt.show()