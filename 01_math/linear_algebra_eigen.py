import numpy as np


def check_eigen_pair(A: np.ndarray, lam: float, v: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Verify A @ v ≈ lam * v.
    Because of floating point error, we use np.allclose with a tolerance.
    """
    left = A @ v
    right = lam * v
    ok = np.allclose(left, right, atol=tol, rtol=tol)
    return ok


def run_case(name: str, A: np.ndarray) -> None:
    print("=" * 60)
    print(f"Case: {name}")
    print("Matrix A:\n", A)

    eigvals, eigvecs = np.linalg.eig(A)

    print("\nEigenvalues (λ):")
    print(eigvals)

    print("\nEigenvectors (each column is a vector v):")
    print(eigvecs)

    print("\nVerify A @ v ≈ λ * v for each eigenpair:")
    for i in range(len(eigvals)):
        lam = eigvals[i]
        v = eigvecs[:, i]
        ok = check_eigen_pair(A, lam, v)

        # Optional: show the two sides for intuition
        left = A @ v
        right = lam * v

        print(f"\n  Pair {i}:")
        print(f"    λ = {lam}")
        print(f"    v = {v}")
        print(f"    A @ v     = {left}")
        print(f"    λ * v     = {right}")
        print(f"    PASS? {ok}")


def main():
    # Case 1: Diagonal matrix (easy to interpret)
    A1 = np.array([
        [2.0, 0.0],
        [0.0, 1.0]
    ])
    run_case("Diagonal (scales x and y differently)", A1)

    # Case 2: Rotation matrix (eigenvalues may be complex if no real eigenvectors)
    theta = np.pi / 4  # 45 degrees
    A2 = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    run_case("Rotation (may produce complex eigenvalues)", A2)

    # Case 3: Shear matrix (upper triangular)
    A3 = np.array([
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    run_case("Shear (can be defective / repeated eigenvalues)", A3)

    print("\nDone. Tips:")
    print("- Try changing matrices and re-running.")
    print("- Observe when eigenvalues/eigenvectors become complex (rotation).")
    print("- For diagonal matrices, eigenvectors often align with coordinate axes.")


if __name__ == "__main__":
    main()
