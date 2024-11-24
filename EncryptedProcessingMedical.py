import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import tenseal as ts
import time
import logging as log
import os

# Set up Kaggle credentials (if not already done)
os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()

# Download the dataset
os.system("kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection --unzip")

print("Dataset downloaded and unzipped.")

# Step 1: Define LeNet-1 Model
class LeNet1(nn.Module):
    def __init__(self):
        super(LeNet1, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2)  # Increase from 4 to 8
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=5, stride=1, padding=0)  # Increase from 12 to 32
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        self.fc1 = nn.Linear(32 * 5 * 5, 2)  # 2 output classes (yes and no)

    def forward(self, x):
        x = self.avgpool1(torch.relu(self.conv1(x)))
        x = self.avgpool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)  # Apply dropout
        x = self.fc1(x)
        return x

# Step 2: Train the model on simulated encrypted data
def quantize(tensor, scale=1000):
    """
    Simulate encryption effects by scaling and rounding tensor values.
    """
    return torch.round(tensor * scale) / scale

def add_noise(tensor, noise_level=0.01):
    """
    Add small random noise to simulate precision loss in encryption.
    """
    noise = torch.randn_like(tensor) * noise_level
    return tensor + noise

# Training the model
def train_model(model, train_loader, epochs, optimizer, criterion):
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for images, labels in train_loader:
            # Simulate encryption effects
            images = quantize(images)  # Simulate encryption effects
            images = add_noise(images)  # Simulate encryption noise

            optimizer.zero_grad()            # Clear previous gradients
            outputs = model(images)          # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()                  # Backward pass
            optimizer.step()                 # Update weights

            running_loss += loss.item()      # Track loss

        # Print loss after every epoch
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

# # Step 3: Initialize Homomorphic Encryption Context
def initialize_bfv():
    context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=786433)
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context

# # Step 4: Encrypt Data
def encrypt_tensor(context, tensor):
    return ts.bfv_vector(context, tensor.flatten().tolist())

# # Step 5: Decrypt Data
def decrypt_tensor(context, encrypted_tensor, shape):
    return torch.tensor(encrypted_tensor.decrypt(), dtype=torch.float).reshape(shape)

# Step 6: Evaluate the Model
def evaluate_model(model, test_loader, context):
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():  # Disable gradient computation for evaluation
        for images, labels in test_loader:
            # Encrypt each image in the batch
            encrypted_images = [encrypt_tensor(context, img) for img in images]
            
            for encrypted_img, label in zip(encrypted_images, labels):
                # Decrypt the encrypted image for inference
                decrypted_img = decrypt_tensor(context, encrypted_img, (3, 28, 28))
                decrypted_img = decrypted_img.unsqueeze(0)  # Add batch dimension

                # Forward pass through the model
                output = model(decrypted_img)
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
    # Define transforms (resize images to 28x28 and normalize)
    transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to 28x28 (LeNet-1 input size)
    transforms.ToTensor(),        # Convert to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])
    # Load the dataset
    dataset = datasets.ImageFolder(root="./brain_tumor_dataset", transform=transform)
    # Split into training and testing sets (80/20 split)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    # Define data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    # Initialize model
    model = LeNet1()
    
    # Define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20 # Number of training epochs

    # Train the model
    print("Starting training...")
    train_model(model, train_loader, epochs, optimizer, criterion)

    # Initialize homomorphic encryption context
    context = initialize_bfv()

    # Evaluate model
    starting_time = time.time()
    accuracy = evaluate_model(model, test_loader, context)
    total_time = time.time() - starting_time
    log.info(f"Accuracy: {accuracy}")
    log.info(f"Time: {total_time}")

if __name__ == "__main__":
    main()
