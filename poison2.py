import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import cg

# Parameters
n = 50  # number of points per side
L = 1.0  # domain length
dx = L / (n - 1)
dy = L / (n - 1)

# Create 2D Laplacian operator with 5-point stencil
k = 1 / dx**2
main_diag = 4 * k * np.ones(n * n)
off_diag = -k * np.ones(n * n - 1)
off_diag2 = -k * np.ones(n * n - n)

# Correct for row boundaries (don't wrap around)
for i in range(1, n):
    off_diag[i * n - 1] = 0

diagonals = [main_diag, off_diag, off_diag, off_diag2, off_diag2]
offsets = [0, -1, 1, -n, n]

A = diags(diagonals, offsets, format="csr")

# Set up the right-hand side f
X, Y = np.meshgrid(np.linspace(0, L, n), np.linspace(0, L, n))

# Example source term: f(x, y) = sin(pi x) sin(pi y)
f = np.sin(np.pi * X) * np.sin(np.pi * Y)
f = f.reshape(n * n)

# Solve the linear system
u, info = cg(A, f, tol=1e-8)

# Reshape the solution back to 2D
U = u.reshape((n, n))

# Save the plot
plt.figure(figsize=(6,5))
plt.imshow(U, extent=[0, L, 0, L], origin='lower', cmap='viridis')
plt.colorbar(label='u(x, y)')
plt.title('2D Poisson Equation Solution with CG')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig("poisson_solution.png", dpi=300)
plt.close()

# Output convergence info
if info == 0:
    print("Conjugate Gradient converged!")
else:
    print(f"Conjugate Gradient did not converge. Info = {info}")

