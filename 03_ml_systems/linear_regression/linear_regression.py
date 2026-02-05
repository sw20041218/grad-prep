import numpy as np

# 1. 生成一组线性数据
np.random.seed(0)

true_w = 2.0
true_b = 1.0

x = np.linspace(0, 10, 50)
noise = np.random.normal(0, 1.0, size=len(x))
y = true_w * x + true_b + noise

# 2. 初始化模型参数
w = 0.0
b = 0.0

lr = 0.01
steps = 1000

# 3. 训练（梯度下降）
for step in range(steps):
    y_pred = w * x + b

    # 均方误差
    loss = np.mean((y_pred - y) ** 2)

    # 对参数求梯度
    dw = np.mean(2 * (y_pred - y) * x)
    db = np.mean(2 * (y_pred - y))

    # 参数更新
    w -= lr * dw
    b -= lr * db

    if step % 100 == 0:
        print(f"step {step:4d} | loss {loss:.4f} | w {w:.4f} | b {b:.4f}")

print("\nTraining finished")
print("Estimated w:", w)
print("Estimated b:", b)
print("True w:", true_w)
print("True b:", true_b)