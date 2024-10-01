import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import pickle

def train_model():
    # Load data
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    # Define hyperparameter grid
    hyperparams_grid = {
        'learning_rate': [0.001, 0.0005],
        'batch_size': [32, 64],
        'num_epochs': [50, 100],
        'dropout_rate': [0.5, 0.3]
    }

    # Initialize variables to store best model
    best_accuracy = 0.0
    best_model_state = None
    best_hyperparams = None

    # Perform hyperparameter tuning using cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for lr in hyperparams_grid['learning_rate']:
        for batch_size in hyperparams_grid['batch_size']:
            for num_epochs in hyperparams_grid['num_epochs']:
                for dropout_rate in hyperparams_grid['dropout_rate']:
                    fold_accuracies = []
                    print(f"Testing hyperparameters: lr={lr}, batch_size={batch_size}, "
                          f"num_epochs={num_epochs}, dropout_rate={dropout_rate}")
                    for train_index, val_index in skf.split(X_train, y_train):
                        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                        # Initialize model
                        input_size = X_train.shape[1]
                        num_classes = len(np.unique(y_train))
                        model = TopologicalPhaseClassifier(input_size, num_classes, dropout_rate)

                        # Loss and optimizer
                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.Adam(model.parameters(), lr=lr)

                        # Training loop
                        for epoch in range(num_epochs):
                            model.train()
                            permutation = torch.randperm(X_train_fold.size()[0])

                            for i in range(0, X_train_fold.size()[0], batch_size):
                                indices = permutation[i:i+batch_size]
                                batch_x, batch_y = X_train_fold[indices], y_train_fold[indices]

                                optimizer.zero_grad()

                                outputs = model(batch_x)
                                loss = criterion(outputs, batch_y)

                                loss.backward()
                                optimizer.step()

                        # Evaluate on validation data
                        model.eval()
                        with torch.no_grad():
                            val_outputs = model(X_val_fold)
                            _, predicted = torch.max(val_outputs.data, 1)
                            total = y_val_fold.size(0)
                            correct = (predicted == y_val_fold).sum().item()
                            val_accuracy = correct / total
                            fold_accuracies.append(val_accuracy)

                    avg_accuracy = np.mean(fold_accuracies)
                    print(f"Average validation accuracy: {avg_accuracy*100:.2f}%")

                    if avg_accuracy > best_accuracy:
                        best_accuracy = avg_accuracy
                        best_model_state = model.state_dict()
                        best_hyperparams = {
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'num_epochs': num_epochs,
                            'dropout_rate': dropout_rate
                        }

    print(f"Best hyperparameters found: {best_hyperparams}")
    print(f"Best cross-validation accuracy: {best_accuracy*100:.2f}%")

    # Train final model with best hyperparameters on full training data
    model = TopologicalPhaseClassifier(input_size, num_classes, best_hyperparams['dropout_rate'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_hyperparams['learning_rate'])

    num_epochs = best_hyperparams['num_epochs']
    batch_size = best_hyperparams['batch_size']

    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(X_train.size()[0])

        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

    # Save the trained model
    torch.save(model.state_dict(), 'topological_phase_classifier.pth')

    # Save best hyperparameters
    with open('best_hyperparams.pkl', 'wb') as f:
        pickle.dump(best_hyperparams, f)

    print("Model training complete. Model saved.")

class TopologicalPhaseClassifier(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.5):
        super(TopologicalPhaseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    train_model()
