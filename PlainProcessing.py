import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import tenseal as ts
import time
import logging as log

# Step 1: Define LeNet-1 Model
class LeNet1(nn.Module):
    def __init__(self):
        super(LeNet1, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2)  # Increase from 4 to 8
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=5, stride=1, padding=0)  # Increase from 12 to 32
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        self.fc1 = nn.Linear(32 * 5 * 5, 10)  # Update fc1 accordingly

    def forward(self, x):
        x = self.avgpool1(torch.relu(self.conv1(x)))
        x = self.avgpool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)  # Apply dropout
        x = self.fc1(x)
        return x

# Step 2: Train the model
# def quantize(tensor, scale=1000):
#     """
#     Simulate encryption effects by scaling and rounding tensor values.
#     """
#     return torch.round(tensor * scale) / scale

# def add_noise(tensor, noise_level=0.01):
#     """
#     Add small random noise to simulate precision loss in encryption.
#     """
#     noise = torch.randn_like(tensor) * noise_level
#     return tensor + noise

# Training the model
def train_model(model, train_loader, epochs, optimizer, criterion):
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for images, labels in train_loader:
            # Simulate encryption effects
            # images = quantize(images)  # Simulate encryption effects
            # images = add_noise(images)  # Simulate encryption noise

            optimizer.zero_grad()            # Clear previous gradients
            outputs = model(images)          # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()                  # Backward pass
            optimizer.step()                 # Update weights

            running_loss += loss.item()      # Track loss

        # Print loss after every epoch
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

# # Step 3: No Homomorphic Encryption Context
# def initialize_bfv():
#     context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=786433)
#     context.generate_galois_keys()
#     context.generate_relin_keys()
#     return context

# # Step 4: Encrypt Data
# def encrypt_tensor(context, tensor):
#     return ts.bfv_vector(context, tensor.flatten().tolist())

# # Step 5: Decrypt Data
# def decrypt_tensor(context, encrypted_tensor, shape):
#     return torch.tensor(encrypted_tensor.decrypt(), dtype=torch.float).reshape(shape)

# Step 6: Evaluate the Model
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():  # Disable gradient computation for evaluation
        for images, labels in test_loader:                        
            # Forward pass through the model
            output = model(images)
            pred = torch.argmax(output, dim=1)
            
            # Compare with true label
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%}")
    return accuracy

# Step 7: Main Function
def main():
    # Load MNIST dataset
    transform = transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Define data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    # Initialize model
    model = LeNet1()
    
    # Define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20 # Number of training epochs

    # Train the model
    print("Starting training...")
    train_model(model, train_loader, epochs, optimizer, criterion)

    # No homomorphic encryption context
    # context = initialize_bfv()

    # Evaluate model
    starting_time = time.time()
    accuracy = evaluate_model(model, test_loader)
    total_time = time.time() - starting_time
    log.info(f"Accuracy: {accuracy}")
    log.info(f"Time: {total_time}")

if __name__ == "__main__":
    main()
