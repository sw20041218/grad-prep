import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ===== 数据 =====
x = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
y = np.sin(x)

# ===== ReLU =====
def relu(z):
    return np.maximum(0, z)

# ===== 网络 =====
class DeepNet:
    def __init__(self, use_residual=False):
        self.use_residual = use_residual
        self.W_in = np.random.randn(1, 16) * 0.5
        self.W = [np.random.randn(16, 16) * 0.5 for _ in range(5)]
        self.W_out = np.random.randn(16, 1) * 0.5

    def forward(self, x):
        h = relu(x @ self.W_in)
        for W in self.W:
            h_new = relu(h @ W)
            if self.use_residual:
                h = h + h_new
            else:
                h = h_new
        return h @ self.W_out

# ===== 数值梯度训练（慢但稳）=====
def train(net, steps=300, lr=1e-3, eps=1e-4):
    losses = []
    for _ in range(steps):
        y_pred = net.forward(x)
        loss = ((y_pred - y) ** 2).mean()
        losses.append(loss)

        # 只对 W_in 做数值梯度（够看趋势）
        grad = np.zeros_like(net.W_in)
        for i in range(net.W_in.shape[0]):
            for j in range(net.W_in.shape[1]):
                net.W_in[i, j] += eps
                loss_pos = ((net.forward(x) - y) ** 2).mean()
                net.W_in[i, j] -= 2 * eps
                loss_neg = ((net.forward(x) - y) ** 2).mean()
                net.W_in[i, j] += eps
                grad[i, j] = (loss_pos - loss_neg) / (2 * eps)

        net.W_in -= lr * grad

    return losses

# ===== 对比 =====
loss_plain = train(DeepNet(use_residual=False))
loss_res = train(DeepNet(use_residual=True))

plt.figure(figsize=(8, 5))
plt.plot(loss_plain, label="Plain Deep Net")
plt.plot(loss_res, label="With Residual")
plt.yscale("log")
plt.xlabel("Step")
plt.ylabel("MSE Loss")
plt.title("Residual Connection Effect (Numerical Grad)")
plt.legend()
plt.show()