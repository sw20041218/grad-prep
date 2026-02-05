import numpy as np


def generate_2d_data(n=300, seed=0):
    """
    生成一个具有强相关性的2D数据云：
    先在一个方向方差大、另一个方向方差小，再做旋转。
    """
    rng = np.random.default_rng(seed)

    # 在“轴对齐坐标系”里：x方差大、y方差小
    x = rng.normal(0, 3.0, size=n)
    y = rng.normal(0, 0.6, size=n)
    X = np.stack([x, y], axis=1)  # (n, 2)

    # 旋转一下，让主方向不是坐标轴，制造“需要PCA”的场景
    theta = np.deg2rad(35)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    X_rot = X @ R.T
    return X_rot


def center_data(X):
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    return Xc, mu


def covariance_matrix(Xc):
    """
    Xc: (n, d) 已中心化数据
    返回协方差矩阵 C: (d, d)
    """
    n = Xc.shape[0]
    C = (Xc.T @ Xc) / (n - 1)
    return C


def pca_from_cov(X, k=1):
    """
    从协方差矩阵做特征分解，返回前k个主成分、特征值、解释方差比、投影结果。
    """
    Xc, mu = center_data(X)
    C = covariance_matrix(Xc)

    # 对称矩阵用 eigh 更稳定（返回升序特征值）
    eigvals, eigvecs = np.linalg.eigh(C)

    # 变成降序：最大特征值在前
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 取前 k 个主成分方向（列向量）
    W = eigvecs[:, :k]          # (d, k)
    lam = eigvals[:k]           # (k,)
    explained_ratio = eigvals / eigvals.sum()  # 全部主成分的解释方差比

    # 投影到前k个主成分上
    Z = Xc @ W                  # (n, k)
    return {
        "X_centered": Xc,
        "mean": mu,
        "cov": C,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "W": W,
        "lam": lam,
        "explained_ratio": explained_ratio,
        "Z": Z,
    }


def main():
    X = generate_2d_data(n=400, seed=42)

    out = pca_from_cov(X, k=1)

    print("=" * 60)
    print("PCA demo (2D -> 1D)")
    print("Data shape X:", X.shape)

    print("\nMean (before centering):", out["mean"].ravel())
    print("\nCovariance matrix C:\n", out["cov"])

    print("\nEigenvalues (descending):", out["eigvals"])
    print("Explained variance ratio:", out["explained_ratio"])

    print("\nTop-1 principal component (direction vector W[:,0]):")
    print(out["W"][:, 0])

    print("\nProject to 1D: Z shape:", out["Z"].shape)
    print("First 10 projected values:\n", out["Z"][:10, 0])

    # 一个小验证：方差 = 最大特征值（在该方向上）
    # Z 是沿主方向的坐标，Z 的样本方差应接近最大特征值
    z_var = out["Z"][:, 0].var(ddof=1)
    print("\nCheck: var(Z) ~ largest eigenvalue")
    print("var(Z) =", z_var)
    print("largest eigenvalue =", out["eigvals"][0])

    print("\nDone. Try:")
    print("- Change rotation angle in generate_2d_data()")
    print("- Change k=2 and see Z become 2D again")
    print("- Temporarily remove centering and observe what breaks")


if __name__ == "__main__":
    main()
