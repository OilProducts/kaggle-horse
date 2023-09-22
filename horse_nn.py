import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import horse_data


def horse_nn_train(job_id):

    X, y, test, test_ids = horse_data.load_and_preprocess()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data into training and validation sets
    X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train_nn.values).float()  # , dtype=torch.float32)
    y_train_tensor = torch.from_numpy(y_train_nn).float()  # , dtype=torch.float32)
    X_val_tensor = torch.from_numpy(X_val_nn.values).float()  # , dtype=torch.float32)
    y_val_tensor = torch.from_numpy(y_val_nn).float()  # , dtype=torch.float32)

    # Create PyTorch DataLoader for batch processing
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=8)


    # Define the neural network
    class NeuralNetwork(nn.Module):
        def __init__(self, input_size):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 96)
            self.fc4 = nn.Linear(96, 96)
            self.fc5 = nn.Linear(96, 48)
            self.fc6 = nn.Linear(48, 3)  # 3 classes for the outcome

        def forward(self, x):
            activation = torch.nn.Tanh()
            x = activation(self.fc1(x))
            x = activation(self.fc2(x))
            x = activation(self.fc3(x))
            x = activation(self.fc4(x))
            x = activation(self.fc5(x))
            x = torch.softmax(self.fc6(x), dim=1)

            return x


    # Instantiate the model
    model = NeuralNetwork(X_train_nn.shape[1])

    print(X_train_tensor)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # Number of epochs
    epochs = 500

    patience = 10
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_train_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(data)

            # Compute the loss
            loss = criterion(outputs, target.long())  # target is of type float, loss expects a long type

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_loader)

        # Optional: Validation loop can be added here

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {average_train_loss:.4f}")

        # Start the validation loop
        model.eval()  # Set the model to evaluation mode
        total_val_loss = 0.0
        correct_predictions = 0

        with torch.no_grad():  # Turn off gradients for validation
            for data, target in val_loader:
                outputs = model(data)
                loss = criterion(outputs, target.long())
                total_val_loss += loss.item()

                _, predicted = outputs.max(1)
                correct_predictions += (predicted == target).sum().item()

        average_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_predictions / len(y_val_nn)

        print(
            f"Epoch [{epoch + 1}/{epochs}], Training Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Stopping early due to lack of validation loss improvement.")
            break

    torch.save(model, f'horse_nn{job_id}.pth')
