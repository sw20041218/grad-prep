import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ===== 数据 =====
x = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
y = np.sin(x)

# ===== 激活函数 =====
def tanh(z):
    return np.tanh(z)

def dtanh(z):
    return 1 - np.tanh(z) ** 2

def relu(z):
    return np.maximum(0, z)

def drelu(z):
    return (z > 0).astype(float)

# ===== 网络 =====
class TinyNet:
    def __init__(self, init="scale", scale=0.1, activation="tanh"):
        h = 16
        if init == "scale":
            self.W1 = scale * np.random.randn(1, h)
            self.W2 = scale * np.random.randn(h, 1)
        elif init == "xavier":
            self.W1 = np.random.randn(1, h) * np.sqrt(1 / 1)
            self.W2 = np.random.randn(h, 1) * np.sqrt(1 / h)
        elif init == "he":
            self.W1 = np.random.randn(1, h) * np.sqrt(2 / 1)
            self.W2 = np.random.randn(h, 1) * np.sqrt(2 / h)

        self.b1 = np.zeros((1, h))
        self.b2 = np.zeros((1, 1))

        self.activation = activation

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        if self.activation == "tanh":
            self.a1 = tanh(self.z1)
        else:
            self.a1 = relu(self.z1)
        return self.a1 @ self.W2 + self.b2

    def backward(self, x, y, y_pred, lr=0.01):
        dy = 2 * (y_pred - y) / len(y)

        dW2 = self.a1.T @ dy
        db2 = dy.sum(axis=0, keepdims=True)

        da1 = dy @ self.W2.T
        if self.activation == "tanh":
            dz1 = da1 * dtanh(self.z1)
        else:
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

# ===== 对比实验 =====
configs = [
    ("scale=0.01", dict(init="scale", scale=0.01)),
    ("scale=0.1", dict(init="scale", scale=0.1)),
    ("scale=1.0", dict(init="scale", scale=1.0)),
    ("Xavier", dict(init="xavier")),
    ("He", dict(init="he")),
]

plt.figure(figsize=(8, 5))

for name, cfg in configs:
    net = TinyNet(**cfg, activation="relu")
    losses = train(net)
    plt.plot(losses, label=name)

plt.yscale("log")
plt.xlabel("Step")
plt.ylabel("MSE Loss")
plt.title("Initialization Comparison (relu)")
plt.legend()
plt.show()