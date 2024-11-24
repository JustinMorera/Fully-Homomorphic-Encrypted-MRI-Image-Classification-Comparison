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
from sklearn.metrics import classification_report, confusion_matrix

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

# Training the model
def train_model(model, train_loader, epochs, optimizer, criterion):
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()            # Clear previous gradients
            outputs = model(images)          # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()                  # Backward pass
            optimizer.step()                 # Update weights

            running_loss += loss.item()      # Track loss

        # Print loss after every epoch
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Step 6: Evaluate the Model
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    model.eval()

    with torch.no_grad():  # Disable gradient computation for evaluation
        for images, labels in test_loader:
            # Forward pass through the model
            output = model(images)
            pred = torch.argmax(output, dim=1)
            
            # Collect predictions and labels
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Compare with true label
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    # Calculate accuracy
    accuracy = correct / total

    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=["No Tumor", "Yes Tumor"], digits=4)
    print("Classification Report:")
    print(report)

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    print(f"Accuracy: {accuracy:.2%}")
    return accuracy, report, cm

# Step 7: Main Function
def main():
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

    epochs = 20  # Number of training epochs

    # Train the model
    print("Starting training...")
    train_model(model, train_loader, epochs, optimizer, criterion)

    # Evaluate model
    print("Evaluating model...")
    starting_time = time.time()
    accuracy, report, cm = evaluate_model(model, test_loader)
    total_time = time.time() - starting_time

    # Log results
    log.info(f"Accuracy: {accuracy}")
    log.info(f"Classification Report:\n{report}")
    log.info(f"Confusion Matrix:\n{cm}")
    log.info(f"Time: {total_time}")
    
if __name__ == "__main__":
    main()
