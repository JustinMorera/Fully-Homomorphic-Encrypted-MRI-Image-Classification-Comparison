import torch
import torch.nn as nn
import time
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score
import numpy as np

# Define LeNet-1 Model
class LeNet(nn.Module):
    def __init__(self, hidden=64, output=10):
        super(LeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = torch.nn.Linear(256, hidden)
        self.fc2 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        x = self.conv1(x)
        # the model uses the square activation function
        x = x * x
        # flattening while keeping the batch axis
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        return x

# Training function
def train_model(model, train_loader, epochs, optimizer, criterion):
    start_train_time = time.time()
    model.train()

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
    end_train_time = time.time()
    print(f"Training Time: {end_train_time - start_train_time:.2f} seconds")
    return end_train_time, start_train_time

# Main Function
def main():
    overall_start_time = time.time()

    # Prepare MNIST data
    transform = transforms.Compose([transforms.ToTensor()])
    # Train data
    train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    # Initialize model
    model = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Training the model...")
    end_train_time, start_train_time = train_model(model, train_loader, epochs=10, optimizer=optimizer, criterion=criterion)

    # Test data
    test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
    test_subset = torch.utils.data.Subset(test_data, range(10000))  # First 10000 samples due to time constraints

    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=1, shuffle=True)

    # Evaluate model
    print("\nEvaluating model...")
    start_eval_time = time.time()
    all_preds = []
    all_labels = []
    all_probs = []
 
    counter = 1
    for image, label in test_loader:
        print(f"Counter: {counter}")
        counter += 1
        output = model(image)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1)

        all_preds.append(pred.item())
        all_labels.append(label.item())
        all_probs.append(probs.squeeze(0).detach().cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    accuracy = (all_preds == all_labels).mean()

    # Additional metrics
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    auroc = roc_auc_score(all_labels, all_probs, multi_class="ovr")

    end_eval_time = time.time()

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"AUROC: {auroc:.4f}")

    total_time = time.time() - overall_start_time

    print("\nEnd-to-End Time Measurements:")
    print(f"Training Time: {end_train_time - start_train_time:.2f} seconds")
    print(f"Evaluation Time: {end_eval_time - start_eval_time:.2f} seconds")
    print(f"Total Time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()