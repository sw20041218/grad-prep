import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ===== 数据 =====
x = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
y = np.sin(x)

# ===== ReLU =====
def relu(z):
    return np.maximum(0, z)

def drelu(z):
    return (z > 0).astype(float)

# ===== BatchNorm（极简版）=====
def batch_norm(z, eps=1e-5):
    mean = z.mean(axis=0, keepdims=True)
    var = z.var(axis=0, keepdims=True)
    return (z - mean) / np.sqrt(var + eps)

# ===== 网络 =====
class TinyNet:
    def __init__(self, use_bn=False):
        h = 16
        scale = 3.0  # 故意设得很大（坏初始化）

        self.W1 = scale * np.random.randn(1, h)
        self.W2 = scale * np.random.randn(h, 1)
        self.b1 = np.zeros((1, h))
        self.b2 = np.zeros((1, 1))

        self.use_bn = use_bn

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        if self.use_bn:
            self.z1 = batch_norm(self.z1)
        self.a1 = relu(self.z1)
        return self.a1 @ self.W2 + self.b2

    def backward(self, x, y, y_pred, lr=0.01):
        dy = 2 * (y_pred - y) / len(y)

        dW2 = self.a1.T @ dy
        db2 = dy.sum(axis=0, keepdims=True)

        da1 = dy @ self.W2.T
        dz1 = da1 * drelu(self.z1)

        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

# ===== 训练 =====
def train(net, steps=2000):
    losses = []
    for _ in range(steps):
        y_pred = net.forward(x)
        loss = ((y_pred - y) ** 2).mean()
        losses.append(loss)
        net.backward(x, y, y_pred)
    return losses

# ===== 对比 =====
net_no_bn = TinyNet(use_bn=False)
net_bn = TinyNet(use_bn=True)

loss_no_bn = train(net_no_bn)
loss_bn = train(net_bn)

plt.figure(figsize=(8, 5))
plt.plot(loss_no_bn, label="No BatchNorm")
plt.plot(loss_bn, label="With BatchNorm")
plt.yscale("log")
plt.xlabel("Step")
plt.ylabel("MSE Loss")
plt.title("Effect of BatchNorm (Bad Initialization)")
plt.legend()
plt.show()