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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score
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
        self.fc1 = nn.Linear(32 * 5 * 5, 2)  # 2 output classes (yes and no)

    def forward(self, x):
        x = self.avgpool1(torch.relu(self.conv1(x)))
        x = self.avgpool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.fc1(x)
        return x

# Step 2: Simulate Encryption Effects
def quantize(tensor, scale=1000):
    return torch.round(tensor * scale) / scale

def add_noise(tensor, noise_level=0.01):
    noise = torch.randn_like(tensor) * noise_level
    return tensor + noise

# Training the model
def train_model(model, train_loader, epochs, optimizer, criterion):
    start_train_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = quantize(images)
            images = add_noise(images)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")
    end_train_time = time.time()
    print(f"Training Time: {end_train_time - start_train_time:.2f} seconds")
    return start_train_time, end_train_time

# Step 3: Initialize Homomorphic Encryption Context
def initialize_bfv():
    start_time = time.time()
    context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=786433)
    context.generate_galois_keys()
    context.generate_relin_keys()
    print(f"Encryption Context Initialization Time: {time.time() - start_time:.2f} seconds")
    return context

# Step 4: Encrypt Data
def encrypt_tensor(context, tensor):
    return ts.bfv_vector(context, tensor.flatten().tolist())

# Step 5: Decrypt Data
def decrypt_tensor(context, encrypted_tensor, shape):
    return torch.tensor(encrypted_tensor.decrypt(), dtype=torch.float).reshape(shape)

# Step 6: Evaluate the Model
def evaluate_model(model, test_loader, context):
    start_eval_time = time.time()
    all_preds = []
    all_labels = []
    all_probs = []
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            encrypted_images = [encrypt_tensor(context, img) for img in images]
            
            for encrypted_img, label in zip(encrypted_images, labels):
                decrypted_img = decrypt_tensor(context, encrypted_img, (3, 28, 28))
                decrypted_img = decrypted_img.unsqueeze(0)

                output = model(decrypted_img)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1)

                all_preds.append(pred.item())
                all_labels.append(label.item())
                all_probs.append(probs.squeeze(0).cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    accuracy = (all_preds == all_labels).mean()

    # Additional metrics
    recall = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")
    precision = precision_score(all_labels, all_preds, average="binary")
    auroc = roc_auc_score(all_labels, all_probs[:, 1])

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["No Tumor", "Tumor"]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"AUROC: {auroc:.4f}")

    end_eval_time = time.time()
    return start_eval_time, end_eval_time

# Step 7: Main Function
def main():
    overall_start_time = time.time()

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.ImageFolder(root="./brain_tumor_dataset", transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    model = LeNet1()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")
    start_train_time, end_train_time = train_model(model, train_loader, epochs=20, optimizer=optimizer, criterion=criterion)

    context = initialize_bfv()

    print("\nEvaluating the model...")
    start_eval_time, end_eval_time = evaluate_model(model, test_loader, context)

    total_time = time.time() - overall_start_time
    print("\nEnd-to-End Time Measurements:")
    print(f"Training Time: {end_train_time - start_train_time:.2f} seconds")
    print(f"Evaluation Time: {end_eval_time - start_eval_time:.2f} seconds")
    print(f"Total Time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()