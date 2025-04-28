import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import cg

# Parameters
n = 50  # grid points per side
L = 1.0  # physical length of the domain
dx = L / (n - 1)
dy = L / (n - 1)

# Create the 2D Laplacian matrix (5-point stencil)
k = 1 / dx**2
main_diag = 4 * k * np.ones(n * n)
off_diag = -k * np.ones(n * n - 1)
off_diag2 = -k * np.ones(n * n - n)

# Correct for row jumps
for i in range(1, n):
    off_diag[i * n - 1] = 0

diagonals = [main_diag, off_diag, off_diag, off_diag2, off_diag2]
offsets = [0, -1, 1, -n, n]

A = diags(diagonals, offsets, format="csr")

# Set up the right-hand side vector b
b = np.zeros(n * n)

# Boundary conditions (Dirichlet)
T_top = 100.0
T_bottom = 0.0
T_left = 0.0
T_right = 0.0

# Apply boundary conditions to b
for i in range(n):
    # Top boundary
    b[i] += T_top * k
    # Bottom boundary
    b[(n-1)*n + i] += T_bottom * k
for j in range(n):
    # Left boundary
    b[j*n] += T_left * k
    # Right boundary
    b[j*n + (n-1)] += T_right * k

# Solve using Conjugate Gradient
x, info = cg(A, b, tol=1e-8)

# Reshape the solution into 2D grid
T = x.reshape((n, n))

# Plot the result
plt.figure(figsize=(6,5))
plt.imshow(T, extent=[0, L, 0, L], origin='lower', cmap='hot')
plt.colorbar(label='Temperature')
plt.title('2D Heat Conduction Solved with CG')
plt.xlabel('x')
plt.ylabel('y')

# Save plot instead of showing
plt.tight_layout()
plt.savefig("heat_conduction_solution.png", dpi=300)
plt.close()  # Close the figure to free memory

# Output convergence info
if info == 0:
    print("Conjugate Gradient converged!")
else:
    print(f"Conjugate Gradient did not converge. Info = {info}")
