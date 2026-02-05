 # Initialization Trick: init scale（初始化尺度）为什么能决定训不训得动？

## 目标
固定任务与网络结构，只改变“权重初始化尺度 scale”，观察：
- tanh 是否更容易进入平台（饱和导致梯度消失）
- ReLU 是否更稳 / 或在过大 scale 下出现不稳定

## 实验设计
- 任务：拟合 y = sin(x)
- 网络：1 hidden layer，H=16
- 激活：tanh vs ReLU
- 变量：init_scale ∈ {0.01, 0.1, 1.0, 3.0}

## 观察重点
1) 哪个 scale 下 loss 下降最快？
2) 哪个 scale 下 tanh 会明显“卡住”（平台）？
3) 哪个 scale 下 ReLU 会出现不稳定（loss 上下乱跳或变大）？

## 关键直觉
- tanh：输入太大 → 输出饱和 → 导数≈0 → 梯度消失
- ReLU：正区间导数=1，但如果大量神经元落在负区间会“死亡”；scale 太大也可能导致更新过猛