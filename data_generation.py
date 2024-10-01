# advanced_data_generation.py

import numpy as np
import json
import os

def complex_to_list(c):
    """Converts a complex number to a list [real, imag] for JSON serialization."""
    return [c.real, c.imag]

def generate_SU2_k_anyon_model(k):
    """
    Generates data for the SU(2)_k anyon model.
    """
    # Anyon types are labeled by j = 0, 1/2, ..., k/2
    anyon_labels = [j/2 for j in range(k+1)]
    d_j = [np.sin(np.pi * (2*j + 1) / (k + 2)) / np.sin(np.pi / (k + 2)) for j in range(k+1)]
    D = np.sqrt(sum([d**2 for d in d_j]))  # Total quantum dimension

    # Compute S matrix
    S_matrix = np.zeros((k+1, k+1), dtype=complex)
    for i in range(k+1):
        for j in range(k+1):
            theta = 2 * np.pi * (2 * anyon_labels[i] + 1) * (2 * anyon_labels[j] + 1) / (2 * (k + 2))
            S_matrix[i, j] = np.sqrt(2 / (k + 2)) * np.sin(theta)

    S_matrix = S_matrix / D  # Normalize

    # Compute T matrix (diagonal elements)
    T_matrix = np.zeros((k+1, k+1), dtype=complex)
    for i in range(k+1):
        h = anyon_labels[i] * (anyon_labels[i] + 1) / (k + 2)  # Conformal weight
        T_matrix[i, i] = np.exp(2j * np.pi * h)

    # Fusion rules: Use Verlinde formula to compute fusion coefficients
    N = np.zeros((k+1, k+1, k+1), dtype=int)
    for i in range(k+1):
        for j in range(k+1):
            for l in range(k+1):
                S_i = S_matrix[i]
                S_j = S_matrix[j]
                S_l = np.conj(S_matrix[l])
                N_ijl = int(round(sum(S_i * S_j * S_l * D)))
                N[i, j, l] = N_ijl

    # Braiding statistics are complex and require more advanced computations (omitted for brevity)

    # Topological Entanglement Entropy
    TEE = np.log(D)

    # Ground State Degeneracy (GSD) on torus: Equal to the number of anyon types
    GSD = k + 1

    # Prepare data for JSON serialization
    S_matrix_serializable = [[complex_to_list(c) for c in row] for row in S_matrix]
    T_matrix_serializable = [complex_to_list(T_matrix[i, i]) for i in range(len(T_matrix))]
    quantum_dimensions = d_j
    total_quantum_dimension = D

    # Create a data dictionary
    data = {
        'anyon_labels': anyon_labels,
        'S_matrix': S_matrix_serializable,
        'T_matrix': T_matrix_serializable,
        'quantum_dimensions': quantum_dimensions,
        'total_quantum_dimension': total_quantum_dimension,
        'TEE': TEE,
        'GSD': GSD,
        'label': f'SU(2)_{k}'
    }

    # Save data to JSON file
    filename = f'SU2_{k}_data.json'
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Data for SU(2)_{k} anyon model saved to '{filename}'.")

def generate_ising_anyon_model():
    """
    Generates data for the Ising anyon model.
    """
    # Anyon types: 1 (vacuum), σ, ψ
    anyon_labels = ['1', 'σ', 'ψ']
    d_j = [1, np.sqrt(2), 1]
    D = np.sqrt(sum([d**2 for d in d_j]))  # Total quantum dimension

    # S matrix
    S_matrix = (1 / D) * np.array([
        [1,        np.sqrt(2), 1],
        [np.sqrt(2), 0,       -np.sqrt(2)],
        [1,       -np.sqrt(2), 1]
    ], dtype=complex)

    # T matrix
    T_matrix = np.diag([
        np.exp(-2j * np.pi * 0 / 16),      # h_1 = 0
        np.exp(-2j * np.pi * 1/16),        # h_σ = 1/16
        np.exp(-2j * np.pi * 1/2)          # h_ψ = 1/2
    ])

    # Fusion rules
    fusion_rules = {
        'σ x σ': ['1', 'ψ'],
        'σ x ψ': 'σ',
        'ψ x ψ': '1'
    }

    # TEE
    TEE = np.log(D)

    # GSD on torus
    GSD = 3

    # Prepare data for JSON serialization
    S_matrix_serializable = [[complex_to_list(c) for c in row] for row in S_matrix]
    T_matrix_serializable = [complex_to_list(T_matrix[i, i]) for i in range(len(T_matrix))]

    # Create a data dictionary
    data = {
        'anyon_labels': anyon_labels,
        'S_matrix': S_matrix_serializable,
        'T_matrix': T_matrix_serializable,
        'quantum_dimensions': d_j,
        'total_quantum_dimension': D,
        'TEE': TEE,
        'GSD': GSD,
        'label': 'Ising'
    }

    # Save data to JSON file
    filename = 'ising_anyon_data.json'
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Data for Ising anyon model saved to '{filename}'.")

def generate_dataset():
    """
    Generates data for multiple anyon models and saves them to JSON files.
    """
    # Create a directory to store data files
    if not os.path.exists('data'):
        os.makedirs('data')
    os.chdir('data')

    # Generate data for SU(2)_k models for k = 2 to 5
    for k in range(2, 6):
        generate_SU2_k_anyon_model(k)

    # Generate data for Ising anyon model
    generate_ising_anyon_model()

    # Generate data for Toric Code and Fibonacci models (from previous script)
    # Implemented here for completeness

    # Toric Code
    generate_toric_code_data()

    # Fibonacci Anyon Model
    generate_fibonacci_anyon_data()

    # Trivial Phase
    generate_trivial_phase_data()

def generate_toric_code_data():
    """
    Generates the S and T matrices for the Toric Code model.
    """
    # Anyon types: 1 (vacuum), e, m, em (e x m)
    # Quantum dimensions: all are 1
    D = 2  # Total quantum dimension

    # S matrix
    S_matrix = (1/D) * np.array([
        [1,  1,  1,  1],
        [1,  1, -1, -1],
        [1, -1,  1, -1],
        [1, -1, -1,  1]
    ])

    # T matrix (trivial for Toric Code)
    T_matrix = np.identity(4)

    # Fusion rules
    fusion_rules = {
        '1 x any': 'any',
        'e x e': '1',
        'm x m': '1',
        'e x m': 'em',
        'e x em': 'm',
        'm x em': 'e',
        'em x em': '1'
    }

    # Braiding statistics (R matrices)
    R_matrices = {
        'e_e': 1,
        'm_m': 1,
        'e_m': -1,
        'm_e': -1,
        'e_em': 1,
        'm_em': 1,
        'em_em': 1
    }

    # Create a data dictionary
    data = {
        'S_matrix': S_matrix.tolist(),
        'T_matrix': T_matrix.tolist(),
        'fusion_rules': fusion_rules,
        'R_matrices': R_matrices,
        'label': 'Toric_Code'
    }

    # Save data to JSON file
    with open('toric_code_data.json', 'w') as f:
        json.dump(data, f, indent=4)

    print("Toric code data saved to 'toric_code_data.json'.")

def generate_fibonacci_anyon_data():
    """
    Generates the S and T matrices for the Fibonacci anyon model.
    """
    # Anyon types: 1 (vacuum), τ
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    D = np.sqrt(1 + phi**2)     # Total quantum dimension

    # S matrix
    S_matrix = (1/D) * np.array([
        [1,    phi],
        [phi, -1]
    ])

    # T matrix (conformal spins h_1 = 0, h_tau = 2/5)
    T_matrix = np.diag([
        np.exp(-2j * np.pi * 0),
        np.exp(-2j * np.pi * 0.2)
    ])

    # Fusion rules
    fusion_rules = {
        '1 x any': 'any',
        'τ x τ': ['1', 'τ']
    }

    # R matrices (braiding phases)
    R_matrices = {
        'τ_τ_1': -np.exp(-4j * np.pi / 5),
        'τ_τ_τ': np.exp(3j * np.pi / 5)
    }

    # Convert complex numbers to [real, imag] for JSON serialization
    def complex_to_list(c):
        return [c.real, c.imag]

    # Prepare data for JSON serialization
    S_matrix_serializable = [[complex_to_list(c) for c in row] for row in S_matrix]
    T_matrix_serializable = [complex_to_list(T_matrix[i, i]) for i in range(len(T_matrix))]
    R_matrices_serializable = {k: complex_to_list(v) for k, v in R_matrices.items()}

    # Create a data dictionary
    data = {
        'S_matrix': S_matrix_serializable,
        'T_matrix': T_matrix_serializable,
        'fusion_rules': fusion_rules,
        'R_matrices': R_matrices_serializable,
        'label': 'Fibonacci'
    }

    # Save data to JSON file
    with open('fibonacci_anyon_data.json', 'w') as f:
        json.dump(data, f, indent=4)

    print("Fibonacci anyon data saved to 'fibonacci_anyon_data.json'.")

def generate_trivial_phase_data():
    """
    Generates the S and T matrices for a trivial (non-topological) phase.
    """
    # Only the vacuum sector
    S_matrix = np.array([[1]])
    T_matrix = np.array([[1]])

    fusion_rules = {
        '1 x 1': '1'
    }

    R_matrices = {
        '1_1': 1
    }

    # Create a data dictionary
    data = {
        'S_matrix': S_matrix.tolist(),
        'T_matrix': T_matrix.tolist(),
        'fusion_rules': fusion_rules,
        'R_matrices': R_matrices,
        'label': 'Trivial'
    }

    # Save data to JSON file
    with open('trivial_phase_data.json', 'w') as f:
        json.dump(data, f, indent=4)

    print("Trivial phase data saved to 'trivial_phase_data.json'.")

if __name__ == "__main__":
    generate_toric_code_data()
    generate_fibonacci_anyon_data()
    generate_trivial_phase_data()


