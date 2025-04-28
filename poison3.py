import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import cg

# Parameters
n_atoms = 100  # number of atoms
k_spring = 10.0  # spring constant

# Construct simple stiffness matrix K (tridiagonal: each atom connected to neighbor)
main_diag = 2 * k_spring * np.ones(n_atoms)
off_diag = -k_spring * np.ones(n_atoms - 1)
diagonals = [main_diag, off_diag, off_diag]
offsets = [0, -1, 1]

K = diags(diagonals, offsets, format="csr")

# External force applied (random small forces)
np.random.seed(42)
f = np.random.randn(n_atoms) * 0.1

# Solve for displacements using CG
x, info = cg(K, f, tol=1e-8)

# Plot displacement profile
plt.figure(figsize=(7,4))
plt.plot(np.linspace(0, 1, n_atoms), x, marker='o')
plt.title('Atom Displacements under Force (CG Solution)')
plt.xlabel('Atom index')
plt.ylabel('Displacement')
plt.grid(True)
plt.tight_layout()
plt.savefig("atom_displacement_cg.png", dpi=300)
plt.close()

# Output convergence info
if info == 0:
    print("Conjugate Gradient converged!")
else:
    print(f"Conjugate Gradient did not converge. Info = {info}")

