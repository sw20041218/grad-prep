import numpy as np

# 定义一个简单的二维函数
# f(x, y) = x^2 + y^2
def f(x, y):
    return x**2 + y**2

# 该函数的梯度（直接给结论，不推导）
def grad_f(x, y):
    return np.array([2*x, 2*y])

# 选一个点
point = np.array([1.0, 2.0])

g = grad_f(point[0], point[1])

print("Point:", point)
print("Gradient at point:", g)

# 随机取几个方向，比较变化率
directions = [
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([1, 1]) / np.sqrt(2),
    g / np.linalg.norm(g)  # 梯度方向
]

print("\nDirectional derivatives (approx):")
for d in directions:
    eps = 1e-4
    x_new = point + eps * d
    delta = f(x_new[0], x_new[1]) - f(point[0], point[1])
    print(f"Direction {d}, change ≈ {delta}")