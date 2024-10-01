# monte_carlo_simulation.py

import numpy as np
import json
import os

def simulate_toric_code(L=10, num_samples=1000, beta=1.0):
    """
    Simulates the toric code model using Monte Carlo methods.

    Args:
        L (int): Linear size of the lattice (LxL).
        num_samples (int): Number of samples to generate.
        beta (float): Inverse temperature for the simulation.
    """
    samples = []
    labels = []

    # Initialize lattice configurations
    for sample_idx in range(num_samples):
        print(f"Simulating sample {sample_idx+1}/{num_samples}")

        # Initialize random spin configurations
        spins = np.random.choice([1, -1], size=(L, L))

        # Monte Carlo steps
        num_steps = L * L * 100  # Adjust as needed for convergence

        for step in range(num_steps):
            # Select a random plaquette
            x = np.random.randint(0, L)
            y = np.random.randint(0, L)

            # Flip spins around the plaquette (to respect gauge invariance)
            spins[x, y] *= -1
            spins[(x + 1) % L, y] *= -1
            spins[x, (y + 1) % L] *= -1
            spins[(x + 1) % L, (y + 1) % L] *= -1

            # Compute energy change (ΔE = 0 for toric code in this setup)
            delta_E = 0

            # Metropolis acceptance criterion
            if delta_E > 0 and np.exp(-beta * delta_E) < np.random.rand():
                # Reject the move
                # Flip back the spins
                spins[x, y] *= -1
                spins[(x + 1) % L, y] *= -1
                spins[x, (y + 1) % L] *= -1
                spins[(x + 1) % L, (y + 1) % L] *= -1

        # Compute Wilson loop operator as a topological invariant
        # For simplicity, compute the product of spins along a loop
        wilson_loop = np.prod(spins[0, :]) * np.prod(spins[:, 0])

        # Prepare sample data
        sample = {
            'L': L,
            'beta': beta,
            'wilson_loop': wilson_loop,
            'spin_configuration': spins.tolist(),
            'label': 'Toric_Code'
        }

        samples.append(sample)
        labels.append('Toric_Code')

    # Save samples to JSON file
    if not os.path.exists('monte_carlo_data'):
        os.makedirs('monte_carlo_data')

    with open('monte_carlo_data/toric_code_samples.json', 'w') as f:
        json.dump(samples, f, indent=4)

    print(f"Simulation complete. Data saved to 'monte_carlo_data/toric_code_samples.json'.")

if __name__ == "__main__":
    simulate_toric_code()
