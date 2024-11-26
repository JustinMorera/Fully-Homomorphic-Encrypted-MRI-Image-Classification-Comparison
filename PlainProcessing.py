import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import time
import logging as log
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, recall_score, precision_score
import numpy as np

# Step 1: Define LeNet-1 Model
class LeNet1(nn.Module):
    def __init__(self):
        super(LeNet1, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=5, stride=1, padding=0)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        x = self.avgpool1(torch.relu(self.conv1(x)))
        x = self.avgpool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x

# Step 2: Training function
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

# Step 3: Noise addition for robustness testing
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
            # Optionally add noise for robustness evaluation
            images = add_noise(images, noise_level=0.01)
            output = model(images)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    auroc = roc_auc_score(all_labels, np.array(all_probs), multi_class="ovr")

    # Generate classification report and confusion matrix
    report = classification_report(all_labels, all_preds, digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"AUROC: {auroc:.4f}")

    return accuracy, recall, f1, auroc, report, cm

# Step 5: Main Function
def main():
    # Load MNIST dataset
    transform = transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

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

    # Evaluate model
    print("\nEvaluating model...")
    start_eval_time = time.time()
    accuracy, recall, f1, auroc, report, cm = evaluate_model(model, test_loader)
    end_eval_time = time.time()

    total_time = end_eval_time - start_train_time

    print("\nEnd-to-End Time Measurements:")
    print(f"Training Time: {end_train_time - start_train_time:.2f} seconds")
    print(f"Evaluation Time: {end_eval_time - start_eval_time:.2f} seconds")
    print(f"Total Time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()