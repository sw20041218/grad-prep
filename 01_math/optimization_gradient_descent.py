import numpy as np

# 一个最简单的函数：f(x) = x^2
def f(x):
    return x**2

# 它的梯度（导数）
def grad_f(x):
    return 2*x

# 梯度下降
x = 5.0              # 起点
lr = 1.1             # 步长（learning rate）
steps = 20

print("Start gradient descent")
print("x    f(x)")

for i in range(steps):
    print(f"{x:.6f}  {f(x):.6f}")
    x = x - lr * grad_f(x)

print("\nFinal x:", x)