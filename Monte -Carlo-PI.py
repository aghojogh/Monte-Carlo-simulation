"""
numpy_project.py
================
A fun NumPy showcase covering three mini-projects:
  1. Monte Carlo estimation of π
  2. 2-D Random Walk simulation
  3. Matrix Algebra playground
"""

import numpy as np

np.random.seed(42)


# ──────────────────────────────────────────────
# 1. MONTE CARLO ESTIMATION OF π
# ──────────────────────────────────────────────
def estimate_pi(n_samples: int = 1_000_000) -> float:
    """
    Throw random points into a 2×2 square centred at the origin.
    Points that land inside the unit circle (r ≤ 1) approximate π/4.
    """
    # Random (x, y) in [-1, 1]
    points = np.random.uniform(-1, 1, size=(n_samples, 2))
    # Distance from origin
    distances = np.linalg.norm(points, axis=1)
    inside = np.sum(distances <= 1.0)
    pi_estimate = 4 * inside / n_samples
    return pi_estimate


print("=" * 50)
print("1. MONTE CARLO π ESTIMATION")
print("=" * 50)
for n in [1_000, 10_000, 100_000, 1_000_000]:
    est = estimate_pi(n)
    error = abs(est - np.pi)
    print(f"  n={n:>10,}  →  π ≈ {est:.6f}  (error: {error:.6f})")
print(f"\n  True PI = {np.pi:.6f}")


# ──────────────────────────────────────────────
# 2. 2-D RANDOM WALK
# ──────────────────────────────────────────────
def random_walk_2d(n_steps: int = 10_000, n_walkers: int = 5):
    """
    Simulate 2-D random walkers using complex numbers as coordinates.

    The key insight: a 2-D point (x, y) is the same as the complex number
    x + y*j. This means:
      - One random step = pick from [right, left, up, down]
                        = pick from [1, -1, 1j, -1j]
      - Distance from origin = abs(complex number)    ← no more np.linalg.norm!
      - Position over time   = np.cumsum(steps)       ← same as before

    Harder to think of, but once you see it — beautifully short to write.
    """
    DIRECTIONS = np.array([1, -1, 1j, -1j])           # the four compass steps

    steps     = np.random.choice(DIRECTIONS, size=(n_steps, n_walkers))
    positions = np.cumsum(steps, axis=0)               # complex positions over time

    final           = positions[-1]                    # where each walker ended up
    final_distances = np.abs(final)                    # distance = magnitude of complex number
    max_distances   = np.abs(positions).max(axis=0)    # furthest point ever reached

    return final, final_distances, max_distances


print("\n" + "=" * 50)
print("2. 2-D RANDOM WALK  (10,000 steps, 5 walkers)")
print("=" * 50)
finals, final_dists, max_dists = random_walk_2d()
for i, (pos, fd, md) in enumerate(zip(finals, final_dists, max_dists)):
    print(f"  Walker {i+1}: final=({pos.real:+6.0f}, {pos.imag:+6.0f})  "
          f"dist from origin={fd:7.2f}  max dist reached={md:7.2f}")

# Theoretical RMS displacement for 2-D walk: sqrt(n_steps)
rms_theory = np.sqrt(10_000)
rms_actual = np.sqrt(np.mean(final_dists**2))
print(f"\n  Theoretical RMS displacement : {rms_theory:.2f}")
print(f"  Simulated  RMS displacement  : {rms_actual:.2f}")


# ──────────────────────────────────────────────
# 3. MATRIX ALGEBRA PLAYGROUND
# ──────────────────────────────────────────────
def matrix_demo(size: int = 4):
    """
    Create a random matrix and explore its properties:
    determinant, inverse, eigenvalues, and a least-squares solve.
    """
    A = np.random.randint(1, 10, size=(size, size)).astype(float)
    b = np.random.randint(1, 20, size=size).astype(float)

    det = np.linalg.det(A)
    inv = np.linalg.inv(A)
    eigenvalues, _ = np.linalg.eig(A)
    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

    return A, b, det, inv, eigenvalues, x


print("\n" + "=" * 50)
print("3. MATRIX ALGEBRA PLAYGROUND  (4×4 random matrix)")
print("=" * 50)
A, b, det, inv, eigs, x = matrix_demo()

print(f"\n  Matrix A:\n{A}")
print(f"\n  Determinant          : {det:.4f}")
print(f"  Rank                 : {np.linalg.matrix_rank(A)}")
print(f"  Condition number     : {np.linalg.cond(A):.4f}")
print(f"  Eigenvalues          : {np.round(eigs.real, 3)}")
print(f"\n  Solving A·x = b  where b = {b}")
print(f"  Solution x           : {np.round(x, 4)}")
print(f"  Verification A·x     : {np.round(A @ x, 4)}")
print(f"  Max absolute error   : {np.max(np.abs(A @ x - b)):.2e}")

# Bonus: SVD decomposition
U, S, Vt = np.linalg.svd(A)
print(f"\n  Singular values (SVD): {np.round(S, 4)}")
print(f"  Reconstruction error : {np.max(np.abs(U @ np.diag(S) @ Vt - A)):.2e}")

print("\n" + "=" * 50)
print("All done.")
print("=" * 50)
