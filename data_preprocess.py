# data_preprocessing.py

import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

def preprocess_data():
    # Load data from all simulations
    data_files = ['kitaev_honeycomb_samples.json', 'toric_code_samples.json', 'fqhe_samples.json']
    data_samples = []
    labels = []

    # Initialize label encoders
    manifold_encoder = LabelEncoder()
    manifold_types = ['plane', 'cylinder', 'torus', 'sphere']
    manifold_encoder.fit(manifold_types)

    for file in data_files:
        with open(f'data/{file}', 'r') as f:
            data = json.load(f)
            for sample in data:
                # Extract features
                energy = sample['energy']
                energy_gap = sample['energy_gap']
                ee_mean = sample['ee_mean']
                ee_std = sample['ee_std']
                chern_number = sample['chern_number']
                system_size = sample['system_size']
                num_particles = sample['num_particles']
                manifold_type = sample['manifold_type']
                params = sample['params']

                # Encode manifold type
                manifold_encoded = manifold_encoder.transform([manifold_type])[0]

                # Combine features into a single vector
                feature_vector = [
                    energy, energy_gap, ee_mean, ee_std, chern_number,
                    system_size, num_particles, manifold_encoded
                ] + params

                data_samples.append(feature_vector)
                labels.append(sample['label'])

    data_samples = np.array(data_samples)

    # Normalize features
    scaler = StandardScaler()
    data_samples = scaler.fit_transform(data_samples)

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Save label names for reference
    label_names = label_encoder.classes_

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data_samples, labels_encoded, test_size=0.2, random_state=42, stratify=labels
    )

    # Save datasets
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

    # Save scaler and encoders
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    with open('manifold_encoder.pkl', 'wb') as f:
        pickle.dump(manifold_encoder, f)
    with open('label_names.pkl', 'wb') as f:
        pickle.dump(label_names, f)

    print("Data preprocessing complete. Datasets saved.")

if __name__ == "__main__":
    preprocess_data()
