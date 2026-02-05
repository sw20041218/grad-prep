import numpy as np

# 二维函数：f(x, y) = x^2 + y^2
def f(v):
    x, y = v
    return x**2 + y**2

# 梯度
def grad_f(v):
    x, y = v
    return np.array([2*x, 2*y])

# 初始点
v = np.array([3.0, 4.0])
lr = 0.1
steps = 10

print("step    x        y        f(x,y)")
for i in range(steps):
    print(f"{i:>2}   {v[0]:>7.4f}  {v[1]:>7.4f}  {f(v):>8.4f}")
    v = v - lr * grad_f(v)

print("\nFinal point:", v)