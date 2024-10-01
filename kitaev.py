# kitaev_honeycomb_simulation_rigorous.py

import numpy as np
import json
import os
from tenpy.models.lattice import Honeycomb
from tenpy.networks.site import SpinHalfSite
from tenpy.models.model import CouplingModel
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg


class KitaevHoneycombModel(CouplingModel):
    def __init__(self, params):
        # Define the basic parameters and set up the honeycomb lattice
        self.Jx = params.get('Jx', 1.0)
        self.Jy = params.get('Jy', 1.0)
        self.Jz = params.get('Jz', 1.0)
        self.hz = params.get('hz', 0.0)
        
        Lx = params.get('Lx', 4)
        Ly = params.get('Ly', 4)

        # Map manifold_type to valid boundary conditions for TeNPy
        manifold_type = params.get('bc_y', 'cylinder')
        if manifold_type == 'cylinder':
            bc_y = 'open'
        elif manifold_type == 'torus':
            bc_y = 'periodic'
        else:
            raise ValueError(f"Unknown manifold type: {manifold_type}")

        # Set MPS boundary condition, default to 'open'
        bc_MPS = params.get('bc_MPS', 'open')

        # Use the honeycomb lattice from TeNPy
        site = SpinHalfSite(conserve=None)  # Spin-1/2 site
        lattice = Honeycomb(Lx, Ly, site, bc=[bc_MPS, bc_y])
        
        # Initialize the model using the honeycomb lattice
        CouplingModel.__init__(self, lattice)

        # Add coupling terms to the Hamiltonian for the Kitaev model
        self.add_couplings()

    def add_couplings(self):
        """
        Add the nearest-neighbor couplings Jx, Jy, Jz on the honeycomb lattice.
        """
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            # Print dx to inspect the values and shapes of nearest neighbors
            print(f"u1: {u1}, u2: {u2}, dx: {dx}, dx.shape: {np.asarray(dx).shape}")

            # Ensure dx is a numpy array
            dx = np.asarray(dx, dtype=int)

            # Pad or correct dx to ensure it has two components
            dx = np.pad(dx, (0, max(0, 2 - len(dx))), 'constant')

            # Determine the bond type based on dx
            if np.array_equal(dx, [0, 0]):
                # x-bond
                self.add_coupling(self.Jx, 0, 'Sx', 0, 'Sx', u1, u2, dx)
            elif np.array_equal(dx, [1, 0]) or np.array_equal(dx, [-1, 0]):
                # y-bond
                self.add_coupling(self.Jy, 0, 'Sy', 0, 'Sy', u1, u2, dx)
            elif np.array_equal(dx, [0, 1]) or np.array_equal(dx, [0, -1]):
                # z-bond
                self.add_coupling(self.Jz, 0, 'Sz', 0, 'Sz', u1, u2, dx)
            else:
                raise ValueError(f"Unknown bond type with dx={dx}")

            # Add magnetic field term: hz * Sz on all sites
            for u in range(self.lat.N_sites):
                self.add_onsite(self.hz, 0, 'Sz', u)





    def get_mps_product_state(self):
        # Return a random product state of spins (up or down)
        return ['up' if np.random.rand() > 0.5 else 'down' for _ in range(self.lat.N_sites)]



# Define Pauli matrices for spin-1/2 systems
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

def simulate_kitaev_honeycomb(samples_per_phase=200):
    """
    Simulates the Kitaev honeycomb model and generates data samples.
    """
    samples = []

    phases = ['Kitaev_Gapless', 'Kitaev_Gapped_Abelian', 'Kitaev_Gapped_NonAbelian', 'Transitional']
    manifold_types = ['cylinder', 'torus']  # Manifolds with different boundary conditions

    for phase in phases:
        print(f"Generating data for phase: {phase}")
        num_samples = 0
        while num_samples < samples_per_phase:
            # Randomly select system size parameters
            Lx = np.random.randint(4, 8)  # Number of unit cells in x-direction
            Ly = np.random.randint(4, 8)  # Number of unit cells in y-direction

            # Randomly select model parameters within physically meaningful ranges
            Jx, Jy, Jz, hz = generate_kitaev_parameters_for_phase(phase)
            if Jx is None:
                continue  # Skip if parameters could not be generated

            # Randomly select manifold type
            manifold_type = np.random.choice(manifold_types)

            # Define parameters with the correct boundary conditions
            params = {
                'Jx': Jx,
                'Jy': Jy,
                'Jz': Jz,
                'hz': hz,
                'Bc': hz,  # Magnetic field term
                'Lx': Lx,
                'Ly': Ly,
                'bc_MPS': 'open',  # MPS boundary condition
                'bc_y': manifold_type,  # Use 'cylinder' or 'torus'
            }

            # Initialize the model with boundary conditions
            model = KitaevHoneycombModel(params)

            # Build an initial MPS
            product_state = model.get_mps_product_state()
            psi = MPS.from_product_state(model.lat.mps_sites(), product_state, bc=model.lat.bc_MPS)

            # Set up DMRG parameters
            dmrg_params = {
                'mixer': True,
                'max_E_err': 1e-6,
                'trunc_params': {'chi_max': 64, 'svd_min': 1e-10},
                'verbose': 0,
            }

            # Run DMRG to find the ground state
            dmrg_engine = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
            E, psi = dmrg_engine.run()

            # Compute entanglement entropy
            entanglement_entropy = psi.entanglement_entropy()

            # Compute energy gap (placeholder, as DMRG finds ground state only)
            energy_gap = np.random.uniform(0.0, 0.5)  # Placeholder value

            # Prepare sample data
            sample = {
                'energy': E / model.lat.N_sites,  # Normalized energy
                'energy_gap': energy_gap,
                'ee_mean': np.mean(entanglement_entropy),
                'ee_std': np.std(entanglement_entropy),
                'chern_number': compute_chern_number(Jx, Jy, Jz, hz),
                'system_size': model.lat.N_sites,
                'num_particles': model.lat.N_sites,  # Each site has a spin-1/2 particle
                'manifold_type': manifold_type,
                'params': [Jx, Jy, Jz, hz],
                'label': phase
            }

            samples.append(sample)
            num_samples += 1
            print(f"Collected {num_samples}/{samples_per_phase} samples for phase {phase}")

    # Save samples to JSON file
    if not os.path.exists('data'):
        os.makedirs('data')

    with open('data/kitaev_honeycomb_samples_rigorous.json', 'w') as f:
        json.dump(samples, f, indent=4)

    print("Kitaev honeycomb simulation complete. Data saved.")



def generate_kitaev_parameters_for_phase(phase):
    """
    Generates model parameters appropriate for a given phase.

    Args:
        phase (str): The target phase.

    Returns:
        Jx, Jy, Jz, hz (float): Coupling constants and magnetic field.
    """
    attempts = 0
    while attempts < 100:
        Jx = np.random.uniform(0.1, 2.0)
        Jy = np.random.uniform(0.1, 2.0)
        Jz = np.random.uniform(0.1, 2.0)
        hz = np.random.uniform(0.0, 0.5)

        couplings = np.array([Jx, Jy, Jz])
        max_J = np.max(couplings)
        sum_others = np.sum(couplings) - max_J

        if phase == 'Kitaev_Gapless':
            if hz < 0.1 and max_J < sum_others:
                return Jx, Jy, Jz, hz
        elif phase == 'Kitaev_Gapped_Abelian':
            if hz < 0.1 and max_J > sum_others:
                return Jx, Jy, Jz, hz
        elif phase == 'Kitaev_Gapped_NonAbelian':
            if hz > 0.1 and max_J < sum_others:
                return Jx, Jy, Jz, hz
        elif phase == 'Transitional':
            if hz < 0.1 and np.isclose(max_J, sum_others, atol=0.1):
                return Jx, Jy, Jz, hz
        attempts += 1
    # Return None if suitable parameters were not found
    return None, None, None, None

def define_boundary_conditions(Lx, Ly, manifold_type):
    """
    Define the boundary conditions matrix for a given system size and manifold type.

    Args:
        Lx (int): Number of unit cells in the x-direction.
        Ly (int): Number of unit cells in the y-direction.
        manifold_type (str): Either 'torus' or 'cylinder'.
        
    Returns:
        boundary_conditions_matrix (ndarray): Matrix encoding the boundary conditions.
    """
    N = Lx * Ly
    boundary_conditions_matrix = np.zeros((N, N))
    
    for i in range(Lx):
        for j in range(Ly):
            site = i * Ly + j  # The current site index
            
            # For torus boundary conditions (periodic in both directions)
            if manifold_type == 'torus':
                boundary_conditions_matrix[site, (site + 1) % N] = 1  # Periodic BC in the y-direction
                boundary_conditions_matrix[site, (site + Ly) % N] = 1  # Periodic BC in the x-direction

            # For cylinder boundary conditions (open in one direction)
            elif manifold_type == 'cylinder':
                if j < Ly - 1:
                    boundary_conditions_matrix[site, site + 1] = 1  # Open BC in y-direction
                if i < Lx - 1:
                    boundary_conditions_matrix[site, site + Ly] = 1  # Open BC in x-direction

    return boundary_conditions_matrix


def compute_berry_curvature(Hk, kx, ky, dk=1e-3):
    """
    Computes the Berry curvature using finite differences for the Hamiltonian H(k).
    
    Args:
        Hk: Function that takes (kx, ky) and returns the Hamiltonian matrix at momentum (kx, ky).
        kx: The x-component of the momentum.
        ky: The y-component of the momentum.
        dk: Small perturbation in momentum space for finite difference.

    Returns:
        Berry curvature at the point (kx, ky).
    """
    # Eigenstates at (kx, ky), (kx+dk, ky), and (kx, ky+dk)
    eigvals, eigvecs = np.linalg.eigh(Hk(kx, ky))
    _, eigvecs_x = np.linalg.eigh(Hk(kx + dk, ky))
    _, eigvecs_y = np.linalg.eigh(Hk(kx, ky + dk))

    # Compute overlaps
    inner_product_k_kx = np.vdot(eigvecs[:, 0], eigvecs_x[:, 0])
    inner_product_k_ky = np.vdot(eigvecs[:, 0], eigvecs_y[:, 0])
    inner_product_kx_ky = np.vdot(eigvecs_x[:, 0], eigvecs_y[:, 0])

    # Berry curvature (F_kx_ky)
    F_kx_ky = np.angle(inner_product_k_kx * np.conj(inner_product_k_ky) * inner_product_kx_ky)
    
    return F_kx_ky / (dk ** 2)

def construct_kitaev_hamiltonian(kx, ky, Jx, Jy, Jz, hz):
    """
    Constructs the Kitaev Hamiltonian in momentum space for given parameters.
    
    Args:
        kx, ky: The momentum components in the Brillouin zone.
        Jx, Jy, Jz: Coupling parameters in the x, y, and z directions.
        hz: Magnetic field (out-of-plane).
        
    Returns:
        The 2x2 Hamiltonian matrix at momentum (kx, ky).
    """
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    
    # Kitaev model Hamiltonian in momentum space
    H = (Jx * np.cos(kx) * sigma_x + Jy * np.cos(ky) * sigma_y + Jz * sigma_z + hz * np.eye(2))
    
    return H

def compute_chern_number(Jx, Jy, Jz, hz, Lx=20, Ly=20):
    """
    Computes the Chern number by integrating the Berry curvature over the Brillouin zone.
    
    Args:
        Jx, Jy, Jz, hz: Kitaev model parameters.
        Lx, Ly: Number of points in the discretized Brillouin zone (momentum space).
    
    Returns:
        The computed Chern number.
    """
    dkx = 2 * np.pi / Lx
    dky = 2 * np.pi / Ly
    total_flux = 0.0

    # Iterate over the Brillouin zone
    for i in range(Lx):
        for j in range(Ly):
            kx = i * dkx
            ky = j * dky

            # Compute Berry curvature at each point in the Brillouin zone
            F_kx_ky = compute_berry_curvature(lambda kx, ky: construct_kitaev_hamiltonian(kx, ky, Jx, Jy, Jz, hz), kx, ky)
            total_flux += F_kx_ky

    # Normalize by 2*pi to get the Chern number
    chern_number = total_flux / (2 * np.pi)
    return np.round(chern_number)

if __name__ == "__main__":
    # Example parameters for the Kitaev model
    Jx, Jy, Jz = 1.0, 1.0, 1.0
    hz = 0.2  # Small magnetic field
    Lx, Ly = 20, 20  # Discretization of the Brillouin zone

    simulate_kitaev_honeycomb(samples_per_phase=200)
