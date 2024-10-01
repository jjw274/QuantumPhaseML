import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse.linalg import lobpcg
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from numpy.linalg import svd
from joblib import Parallel, delayed
from tqdm import tqdm  
import json

def site_index(x, y, sublattice, Lx, Ly):
    """
    Computes the site index for a given x, y, sublattice in a lattice of size Lx, Ly.
    """
    return (x % Lx) + Lx * (y % Ly) + sublattice * Lx * Ly

def create_warped_lattice(Lx, Ly, t1, t2, phi, manifold, geometry='honeycomb', warp_type=None, warp_strength=0.0):
    """
    Creates a lattice (honeycomb, square, triangular) with optional warping.
    
    Parameters:
    - Lx, Ly: dimensions of the lattice
    - t1, t2: nearest and next-nearest neighbor hopping terms
    - phi: phase for the next-nearest neighbor hopping
    - manifold: 'torus' or 'cylinder' (boundary conditions)
    - geometry: 'honeycomb', 'square', 'triangular'
    - warp_type: Type of warping ('parabolic', 'spherical', etc.)
    - warp_strength: Strength of the warping effect
    
    Returns:
    - Hamiltonian matrix in sparse format
    - List of positions of the sites
    """
    num_sites = 2 * Lx * Ly if geometry == 'honeycomb' else Lx * Ly
    H = lil_matrix((num_sites, num_sites), dtype=complex)
    positions = []

    # Define lattice vectors based on geometry
    if geometry == 'honeycomb':
        a1 = np.array([np.sqrt(3), 0])
        a2 = np.array([np.sqrt(3) / 2, 3 / 2])
        delta = [
            np.array([0, 1]),
            np.array([-np.sqrt(3) / 2, -0.5]),
            np.array([np.sqrt(3) / 2, -0.5])
        ]
        nn_delta = [
            np.array([np.sqrt(3), 0]),
            np.array([-np.sqrt(3) / 2, 1.5]),
            np.array([-np.sqrt(3) / 2, -1.5])
        ]
    elif geometry == 'square':
        a1 = np.array([1, 0])
        a2 = np.array([0, 1])
        delta = [
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([-1, 0]),
            np.array([0, -1])
        ]
        nn_delta = [
            np.array([1, 1]),
            np.array([1, -1]),
            np.array([-1, 1]),
            np.array([-1, -1])
        ]
    elif geometry == 'triangular':
        a1 = np.array([1, 0])
        a2 = np.array([0.5, np.sqrt(3) / 2])
        delta = [
            np.array([1, 0]),
            np.array([0.5, np.sqrt(3) / 2]),
            np.array([-0.5, np.sqrt(3) / 2])
        ]
        nn_delta = [
            np.array([1.5, np.sqrt(3) / 2]),
            np.array([-1.5, np.sqrt(3) / 2]),
            np.array([1.5, -np.sqrt(3) / 2])
        ]
    else:
        raise ValueError("Unknown lattice geometry")

    # Apply warping function to the positions
    def apply_warping(x, y):
        """Apply lattice warping based on type."""
        if warp_type == 'parabolic':
            # Parabolic warping
            x_new = x + warp_strength * (x**2)
            y_new = y + warp_strength * (y**2)
        elif warp_type == 'spherical':
            # Spherical warping approximation
            R = np.sqrt(Lx**2 + Ly**2) / 2  # Approximate radius
            theta_x = x / R
            theta_y = y / R
            x_new = R * np.sin(theta_x)
            y_new = R * np.sin(theta_y)
        else:
            # No warping applied
            x_new, y_new = x, y
        return x_new, y_new

    # Build the Hamiltonian
    for x in range(Lx):
        for y in range(Ly):
            for sublattice in [0, 1] if geometry == 'honeycomb' else [0]:
                idx = site_index(x, y, sublattice, Lx, Ly)
                
                # Apply warping to the lattice positions
                x_warped, y_warped = apply_warping(x, y)
                positions.append((x_warped, y_warped, sublattice))

                # Nearest-neighbor hopping
                for d in delta:
                    if sublattice == 0 or geometry != 'honeycomb':
                        nx = x + d[0]
                        ny = y + d[1]
                        if manifold == 'torus' or (0 <= ny < Ly):
                            n_idx = site_index(int(nx), int(ny), 1 if geometry == 'honeycomb' else 0, Lx, Ly)
                            H[idx, n_idx] = -t1
                    else:
                        continue

                # Next-nearest-neighbor hopping
                for d in nn_delta:
                    nx = x + d[0]
                    ny = y + d[1]
                    if manifold == 'torus' or (0 <= ny < Ly):
                        n_idx = site_index(int(nx), int(ny), sublattice, Lx, Ly)
                        H[idx, n_idx] += -t2 * np.exp(1j * phi * np.sign(d[1]))

    return H.tocsr(), positions


def berry_curvature(U1, U2, U3, U4):
    F12 = np.dot(U1.conj().T, U2)
    F23 = np.dot(U2.conj().T, U3)
    F34 = np.dot(U3.conj().T, U4)
    F41 = np.dot(U4.conj().T, U1)

    F = np.dot(np.dot(F12, F23), np.dot(F34, F41))
    curvature = np.angle(np.linalg.det(F))

    # Ignore extremely small curvature values (set a tolerance)
    if np.abs(curvature) < 1e-10:
        return 0.0
    return curvature


def H_func(k, Lx, Ly, t1, t2, phi, manifold):
    """
    Generate the Hamiltonian for a honeycomb lattice in k-space (kx, ky) with 
    nearest- and next-nearest-neighbor hopping, considering periodic boundary 
    conditions on a torus or open boundary conditions on a cylinder.

    Parameters:
    k : array-like
        2D vector for kx, ky.
    Lx, Ly : int
        Size of the lattice in x and y directions.
    t1 : float
        Nearest-neighbor hopping strength.
    t2 : float
        Next-nearest-neighbor hopping strength.
    phi : float
        Phase factor for the next-nearest-neighbor hopping.
    manifold : str
        'torus' for periodic boundary conditions or 'cylinder' for open boundary conditions.

    Returns:
    H : scipy.sparse.csr_matrix
        The Hamiltonian matrix in sparse format.
    """
    # Define nearest-neighbor vectors (between sublattices A and B)
    delta = [
        np.array([0, 1]),                                # Upward (0,1) direction
        np.array([-np.sqrt(3)/2, -0.5]),                 # Down-left (-√3/2, -0.5)
        np.array([np.sqrt(3)/2, -0.5])                   # Down-right (√3/2, -0.5)
    ]

    # Define next-nearest-neighbor vectors (within sublattices A and B)
    nn_delta = [
        np.array([np.sqrt(3), 0]),                       # Right (√3, 0)
        np.array([-np.sqrt(3)/2, 1.5]),                  # Up-left (-√3/2, 1.5)
        np.array([-np.sqrt(3)/2, -1.5])                  # Down-left (-√3/2, -1.5)
    ]

    # Hamiltonian matrix with 2 sublattices, total number of sites is 2 * Lx * Ly
    num_sites = 2 * Lx * Ly
    H = lil_matrix((num_sites, num_sites), dtype=complex)

    # Helper function for site indexing
    def site_index(x, y, sublattice, Lx, Ly):
        """Helper function to compute the site index for sublattice (A=0, B=1)."""
        return (x % Lx) + Lx * (y % Ly) + sublattice * Lx * Ly

    # Extract kx, ky from k-space vector
    kx, ky = k

    # Loop over lattice sites
    for x in range(Lx):
        for y in range(Ly):
            for sublattice in [0, 1]:
                idx = site_index(x, y, sublattice, Lx, Ly)

                # Nearest-neighbor hopping (A to B and B to A)
                for d in delta:
                    if sublattice == 0:  # Hopping from A to B
                        nx = int((x + d[0]) % Lx)  # Ensure nx is an integer
                        ny = int((y + d[1]) % Ly)  # Ensure ny is an integer
                        if manifold == 'torus' or (0 <= ny < Ly):
                            n_idx = site_index(nx, ny, 1, Lx, Ly)
                            phase_factor = np.exp(1j * (kx * d[0] + ky * d[1]))
                            H[idx, n_idx] = -t1 * phase_factor
                    elif sublattice == 1:  # Optionally include reverse hopping from B to A
                        nx = int((x - d[0]) % Lx)  # Ensure nx is an integer
                        ny = int((y - d[1]) % Ly)  # Ensure ny is an integer
                        if manifold == 'torus' or (0 <= ny < Ly):
                            n_idx = site_index(nx, ny, 0, Lx, Ly)
                            phase_factor = np.exp(-1j * (kx * d[0] + ky * d[1]))
                            H[idx, n_idx] = -t1 * phase_factor

                # Next-nearest-neighbor hopping (within sublattice)
                for d in nn_delta:
                    nx = int((x + d[0]) % Lx)  # Ensure nx is an integer
                    ny = int((y + d[1]) % Ly)  # Ensure ny is an integer
                    if manifold == 'torus' or (0 <= ny < Ly):
                        n_idx = site_index(nx, ny, sublattice, Lx, Ly)
                        phase_factor = np.exp(1j * (kx * d[0] + ky * d[1]))
                        # Apply the phase `phi` for next-nearest-neighbor hopping
                        H[idx, n_idx] += -t2 * np.exp(1j * phi * np.sign(d[1])) * phase_factor

    return H.tocsr()  # Return sparse matrix in CSR format for efficiency

from scipy.sparse.linalg import lobpcg

def compute_chern_number(Lx, Ly, t1, t2, phi, manifold, num_bands):
    """
    Computes the Chern number using the method of discretized Berry curvature.
    This version computes it sequentially without parallel processing.
    """
    # Define k-grid
    kx_vals = np.linspace(-np.pi, np.pi, Lx * 10)  # Increase k-space resolution
    ky_vals = np.linspace(-np.pi, np.pi, Ly * 10)

    total_chern = 0  # Initialize the total Berry curvature

    # Sequential calculation over plaquettes
    for i in range(len(kx_vals) - 1):
        for j in range(len(ky_vals) - 1):
            # Compute the Berry curvature for each plaquette
            berry_curv = compute_single_plaquette(i, j, kx_vals, ky_vals, Lx, Ly, t1, t2, phi, manifold, num_bands)
            total_chern += berry_curv

    # Normalize by the area of the Brillouin zone
    chern_number = total_chern / (2 * np.pi)
    print(f"Total Berry Curvature: {total_chern}")  # Add print statement
    print(f"Chern Number (before rounding): {chern_number}")

    # Round to nearest integer (to account for numerical errors)
    return np.round(chern_number)

def compute_single_plaquette(i, j, kx_vals, ky_vals, Lx, Ly, t1, t2, phi, manifold, num_bands):
    k1 = np.array([kx_vals[i], ky_vals[j]])
    k2 = np.array([kx_vals[i+1], ky_vals[j]])
    k3 = np.array([kx_vals[i+1], ky_vals[j+1]])
    k4 = np.array([kx_vals[i], ky_vals[j+1]])

    Hk1 = H_func(k1, Lx, Ly, t1, t2, phi, manifold)
    Hk2 = H_func(k2, Lx, Ly, t1, t2, phi, manifold)
    Hk3 = H_func(k3, Lx, Ly, t1, t2, phi, manifold)
    Hk4 = H_func(k4, Lx, Ly, t1, t2, phi, manifold)

    # Make sure we capture more bands, using at least `Lx` bands for a larger system
    num_bands = min(num_bands, Hk1.shape[0])  # Ensure num_bands doesn't exceed system size

    # Use dense solver eigh for smaller systems
    # Compute eigenvalues and eigenvectors at k1
    eigvals1, eigvecs1 = eigh(Hk1.toarray())
    idx1 = np.argsort(eigvals1)
    eigvecs1 = eigvecs1[:, idx1]  # Sort eigenvectors according to sorted eigenvalues

    # Compute eigenvalues and eigenvectors at k2
    eigvals2, eigvecs2 = eigh(Hk2.toarray())
    idx2 = np.argsort(eigvals2)
    eigvecs2 = eigvecs2[:, idx2]  # Sort eigenvectors according to sorted eigenvalues

    # Compute eigenvalues and eigenvectors at k3
    eigvals3, eigvecs3 = eigh(Hk3.toarray())
    idx3 = np.argsort(eigvals3)
    eigvecs3 = eigvecs3[:, idx3]  # Sort eigenvectors according to sorted eigenvalues

    # Compute eigenvalues and eigenvectors at k4
    eigvals4, eigvecs4 = eigh(Hk4.toarray())
    idx4 = np.argsort(eigvals4)
    eigvecs4 = eigvecs4[:, idx4]  # Sort eigenvectors according to sorted eigenvalues


    return berry_curvature(eigvecs1, eigvecs2, eigvecs3, eigvecs4)


def compute_entanglement_entropy(eigenvectors, partition_size, manifold):
    """
    Computes the entanglement entropy of the ground state.
    
    Parameters:
    - eigenvectors: eigenvectors of the Hamiltonian (ground state)
    - partition_size: number of sites in one partition (for bipartitioning)
    - manifold: 'torus' or 'cylinder'
    
    Returns:
    - entropy_mean: mean entanglement entropy
    """
    n_sites = eigenvectors.shape[0]
    
    if partition_size <= 0 or partition_size >= n_sites:
        print(f"Warning: Invalid partition size {partition_size} for n_sites {n_sites}")
        return np.nan, np.nan

    if manifold == 'torus':
        reshaped_state = eigenvectors[:, 0].reshape(partition_size, n_sites // partition_size)
    elif manifold == 'cylinder':
        # Only use the first `partition_size` rows for open boundary
        reshaped_state = eigenvectors[:, 0][:partition_size].reshape(partition_size, -1)
    
    # Catch errors in SVD if reshaping fails
    try:
        _, singular_vals, _ = svd(reshaped_state, full_matrices=False)
    except np.linalg.LinAlgError as e:
        print(f"SVD computation failed: {e}")
        return np.nan, np.nan

    singular_vals = singular_vals**2
    singular_vals = singular_vals[singular_vals > 1e-12]  # Filter out very small values

    if len(singular_vals) == 0:
        print("Warning: Singular values are all too small.")
        return np.nan, np.nan

    entropy = -np.sum(singular_vals * np.log(singular_vals))
    return entropy, np.std(singular_vals)

def simulate_sample(Lx, Ly, phase, manifold):
    """
    Simulates a single sample of the fractional quantum Hall state with random lattice geometry and warping.
    
    Parameters:
    - Lx, Ly: lattice dimensions
    - phase: one of the specified phases
    - manifold: 'torus' or 'cylinder'
    
    Returns:
    - sample_data: dictionary containing the computed features
    """
    print(f"Simulating Lx={Lx}, Ly={Ly}, phase={phase}, manifold={manifold}")
    # Define parameters for each phase
    if phase == 'Laughlin_1/3':
        t1 = 1.0
        t2 = 0.1
        phi = np.pi / 2
    elif phase == 'Moore_Read':
        t1 = 1.0
        t2 = 0.2
        phi = np.pi / 3
    elif phase == 'Pfaffian':
        t1 = 1.0
        t2 = 0.3
        phi = np.pi / 4
    elif phase == 'Halperin_331':
        t1 = 1.0
        t2 = 0.4
        phi = np.pi / 5
    elif phase == 'Composite_Fermion':
        t1 = 1.0
        t2 = 0.5
        phi = np.pi / 6
    elif phase == 'transitional':
        t1 = 1.0
        t2 = np.random.uniform(0, 0.5)
        phi = np.random.uniform(0, np.pi)
    else:
        raise ValueError("Unknown phase specified.")
    
    # Randomly choose geometry and warping parameters
    geometries = ['honeycomb', 'square', 'triangular']
    warp_types = [None, 'parabolic', 'spherical']
    geometry = np.random.choice(geometries)
    warp_type = np.random.choice(warp_types)
    warp_strength = np.random.uniform(0, 0.2) if warp_type is not None else 0.0

    # Create the lattice with geometry and warping
    H, positions = create_warped_lattice(Lx, Ly, t1, t2, phi, manifold, geometry=geometry, warp_type=warp_type, warp_strength=warp_strength)

    print(f"Constructed Hamiltonian for Lx={Lx}, Ly={Ly}, manifold={manifold}, geometry={geometry}, warp_type={warp_type}")
    
    H = H.tocsr()

    try:
        eigvals, eigvecs = eigsh(H, k=min(Lx * Ly, H.shape[0]), which='SM', maxiter=20000, tol=1e-8)
    except Exception as e:
        print(f"eigsh failed, falling back to dense solver: {e}")
        eigvals, eigvecs = np.linalg.eigh(H.toarray())

    print(f"Computed eigenvalues for Lx={Lx}, Ly={Ly}, manifold={manifold}")

    normalized_energy = np.mean(eigvals) / np.max(np.abs(eigvals))

    # Ensure that we do not access out-of-bound indices in eigvals
    if len(eigvals) > Lx * Ly:
        energy_gap = eigvals[Lx * Ly] - eigvals[Lx * Ly - 1]
    else:
        energy_gap = 0  # Set a default value or handle this case as needed

    print("Calculated energy gap")

    divisors = [i for i in range(1, (Lx*Ly) + 1) if (Lx*Ly) % i == 0]

    valid_divisors = [d for d in divisors if 2 <= d <= (Lx*Ly) // 2]

    if len(valid_divisors) == 0:
        valid_divisors = [Lx]
    
    partition_size = np.random.choice(valid_divisors)

    if len(eigvecs) >= Lx:
        entropy_mean, entropy_std = compute_entanglement_entropy(eigvecs[:, :Lx*Ly], partition_size=partition_size, manifold=manifold)
    else:
        entropy_mean, entropy_std = np.nan, np.nan

    print("Calculated Entropy")

    chern_number = compute_chern_number(
        Lx=Lx, 
        Ly=Ly, 
        t1=t1, 
        t2=t2, 
        phi=phi, 
        manifold=manifold, 
        num_bands=Lx
    )

    # Include geometry and warping in the sample data
    sample_data = {
        'normalized_energy': normalized_energy.real,
        'energy_gap': energy_gap.real,
        'entanglement_entropy_mean': entropy_mean,
        'entanglement_entropy_std': entropy_std,
        'chern_number': chern_number,
        'system_size': (Lx, Ly),
        'manifold_type': manifold,
        'phase': phase,
        'partition_size': partition_size,
        'geometry': geometry,
        'warp_type': warp_type,
        'warp_strength': warp_strength
    }

    return sample_data


def convert_to_serializable(data):
    """
    Converts NumPy types to Python native types for JSON serialization.
    Recursively traverses lists and dictionaries to convert all values.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.int64, np.int32)):  # or other numpy integer types
        return int(data)
    elif isinstance(data, (np.float64, np.float32)):  # or other numpy float types
        return float(data)
    elif isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(i) for i in data]
    else:
        return data

import os

def main():
    # Define system sizes and other parameters
    system_sizes = [(2, 2), (4, 4), (6, 6), (8, 8), (10, 10)]
    num_samples = 200
    manifolds = ['torus', 'cylinder']
    phases = ['Laughlin_1/3', 'Moore_Read', 'Pfaffian', 'Halperin_331', 'Composite_Fermion', 'transitional']

    # Create parameter combinations
    param_combinations = [
        (Lx, Ly, phase, manifold)
        for Lx, Ly in system_sizes
        for phase in phases
        for manifold in manifolds
        for _ in range(num_samples)
    ]

    # Start writing to the JSON file and initiate the JSON array
    with open('fqhe_samples.json', 'a', buffering=1) as f:  # Use 'a' for append mode and line-buffered I/O
        f.write('[\n')  # Start the JSON array

    # Parallel processing using joblib for simulations
    for idx, result in enumerate(Parallel(n_jobs=-1)(
            delayed(simulate_sample)(Lx, Ly, phase, manifold)
            for Lx, Ly, phase, manifold in tqdm(param_combinations, desc="Simulating", total=len(param_combinations))
    )):
        # Convert result to a JSON serializable format
        result_serializable = convert_to_serializable(result)

        # Append the result to the file
        with open('fqhe_samples.json', 'a', buffering=1) as f:
            json.dump(result_serializable, f, indent=4)
            if idx < len(param_combinations) - 1:
                f.write(',\n')  # Add a comma between JSON objects for proper formatting

            f.flush()  # Flush the buffer to ensure the data is written to disk
            os.fsync(f.fileno())  # Ensure the OS flushes the buffer to the file system

    # Finish the JSON array after all simulations are done
    with open('fqhe_samples.json', 'a', buffering=1) as f:
        f.write('\n]')  # Close the JSON array

    print("Results saved to fqhe_samples.json")

if __name__ == "__main__":
    main()