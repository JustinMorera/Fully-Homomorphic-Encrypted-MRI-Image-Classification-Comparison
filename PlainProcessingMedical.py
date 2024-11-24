import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import time
import logging as log
import os
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, roc_auc_score, precision_score
import numpy as np

# Set up Kaggle credentials (if not already done)
os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()

# Download the dataset
os.system("kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection --unzip")
print("Dataset downloaded and unzipped.")

# Step 1: Define LeNet-1 Model
class LeNet1(nn.Module):
    def __init__(self):
        super(LeNet1, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=5, stride=1, padding=0)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 5 * 5, 2)  # 2 output classes (No Tumor, Yes Tumor)

    def forward(self, x):
        x = self.avgpool1(torch.relu(self.conv1(x)))
        x = self.avgpool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x

# Step 2: Training the model
def train_model(model, train_loader, epochs, optimizer, criterion):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Step 3: Add noise to inputs (for robustness testing)
def add_noise(images, noise_level=0.01):
    noise = torch.randn_like(images) * noise_level
    return images + noise

# Step 4: Evaluate the Model
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = add_noise(images, noise_level=0.01)  # Add small noise
            output = model(images)
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    auroc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])

    # Generate classification report and confusion matrix
    report = classification_report(all_labels, all_preds, target_names=["No Tumor", "Yes Tumor"], digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")

    return accuracy, recall, f1, auroc, report, cm

# Step 5: Main Function
def main():
    # Define transforms (resize images to 28x28 and normalize)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to 28x28 (LeNet-1 input size)
        transforms.ToTensor(),        # Convert to PyTorch tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Load the dataset
    dataset = datasets.ImageFolder(root="./brain_tumor_dataset", transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    # Initialize model
    model = LeNet1()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 20

    # Train the model
    print("Starting training...")
    start_train_time = time.time()
    train_model(model, train_loader, epochs, optimizer, criterion)
    end_train_time = time.time()

    # Evaluate the model
    print("\nEvaluating model...")
    start_eval_time = time.time()
    accuracy, recall, f1, auroc, report, cm = evaluate_model(model, test_loader)
    end_eval_time = time.time()

    # Print timing metrics
    total_time = end_eval_time - start_train_time
    print("\nEnd-to-End Time Measurements:")
    print(f"Training Time: {end_train_time - start_train_time:.2f} seconds")
    print(f"Evaluation Time: {end_eval_time - start_eval_time:.2f} seconds")
    print(f"Total Time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()