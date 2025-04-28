import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Lennard-Jones potential and force
def lj_potential(r):
    return 4 * ((1/r)**12 - (1/r)**6)

def lj_force(r):
    return 24 * (2*(1/r)**13 - (1/r)**7)

# Total potential energy of the system
def total_energy(x):
    N = len(x)
    energy = 0.0
    for i in range(N):
        for j in range(i+1, N):
            r = np.abs(x[i] - x[j])
            if r != 0:
                energy += lj_potential(r)
    return energy

# Gradient (negative forces)
def total_gradient(x):
    N = len(x)
    grad = np.zeros_like(x)
    for i in range(N):
        for j in range(N):
            if i != j:
                r = x[i] - x[j]
                dist = np.abs(r)
                if dist != 0:
                    grad[i] += lj_force(dist) * np.sign(r)
    return grad

# Number of particles
N_particles = 5

# Initial random positions
np.random.seed(42)
x0 = np.random.rand(N_particles) * 3.0  # spread out to avoid huge forces

# Minimize energy using Conjugate Gradient
res = minimize(total_energy, x0, method='CG', jac=total_gradient, tol=1e-6)

# Final relaxed positions
x_relaxed = res.x

# Plot initial and relaxed positions
plt.figure(figsize=(8, 2))
plt.scatter(x0, np.zeros_like(x0), label='Initial', color='red', s=100)
plt.scatter(x_relaxed, np.zeros_like(x_relaxed), label='Relaxed', color='blue', s=100)
plt.yticks([])
plt.legend()
plt.title('Relaxation of Particles under Lennard-Jones Potential (CG)')
plt.xlabel('Position')
plt.grid(True)
plt.tight_layout()
plt.savefig("lj_relaxation_cg.png", dpi=300)
plt.close()

# Output results
print("Initial Positions:", x0)
print("Relaxed Positions:", x_relaxed)
print(f"Final Total Energy: {total_energy(x_relaxed):.6f}")

