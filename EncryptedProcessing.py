import tenseal as ts
import torch
import torch.nn as nn
import time
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score
import numpy as np

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

class EncLeNet:
    def __init__(self, torch_nn):
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
        
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()
        
        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()      
        
    def forward(self, enc_image, num_windows):
        # conv layer
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_image.conv2d_im2col(kernel, num_windows) + bias
            enc_channels.append(y)
        # pack all channels into a single flattened vector
        enc_image = ts.CKKSVector.pack_vectors(enc_channels)
        # square activation
        enc_image.square_()
        # fc1 layer
        enc_image = enc_image.mm(self.fc1_weight) + self.fc1_bias
        # square activation
        enc_image.square_()
        # fc2 layer
        enc_image = enc_image.mm(self.fc2_weight) + self.fc2_bias
        return enc_image
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

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

def train_model(model, train_loader, epochs, optimizer, criterion):
    start_train_time = time.time()
    model.train()
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
    return end_train_time, start_train_time

def main():
    overall_start_time = time.time()

    # Initialize BFV context
    print("Generating encryption keys and context...")
    bits_scale = 26
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31])
    # set the scale
    context.global_scale = pow(2, bits_scale)
    # galois keys are required to do ciphertext rotations
    context.generate_galois_keys()

    # Prepare MNIST data
    transform = transforms.Compose([transforms.ToTensor()])
    # Train data
    train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    # Initialize plaintext model
    plaintext_model = LeNet()  # Use unencrypted layers for training
    
    # Train the model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(plaintext_model.parameters(), lr=0.001)
    print("Training the model on encryption-simulated data...")
    end_train_time, start_train_time = train_model(plaintext_model, train_loader, epochs=10, optimizer=optimizer, criterion=criterion)
        
    # Test data
    test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
    test_subset = torch.utils.data.Subset(test_data, range(10000))  # First 10000 samples due to time constraints

    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=1, shuffle=True)
    encrypted_model = EncLeNet(plaintext_model)

    # Encryption and Evaluation
    print("Encrypting and evaluating data on the model...")
    start_eval_time = time.time()
    all_preds = []
    all_labels = []
    all_probs = []

    running_encryption_time = 0
    running_decryption_time = 0
    kernel_shape = plaintext_model.conv1.kernel_size
    stride = plaintext_model.conv1.stride[0]
 
    counter = 1
    for image, label in test_loader:
        # Encoding and encryption
        encryption_start_time = time.time()
        enc_image, num_windows = ts.im2col_encoding(
            context, image.view(28, 28).tolist(), kernel_shape[0],
            kernel_shape[1], stride
        )
        encryption_end_time = time.time()
        running_encryption_time += encryption_end_time - encryption_start_time
        print(f"Counter: {counter}")
        counter += 1
        # Encrypted inference/evaluation
        encrypted_output = encrypted_model(enc_image, num_windows)
        start_decrypt_time = time.time()
        decrypted_output = encrypted_output.decrypt()
        end_decrypt_time = time.time()
        running_decryption_time += end_decrypt_time - start_decrypt_time

        probs = torch.softmax(torch.tensor(decrypted_output), dim=0)
        pred = torch.argmax(probs, dim=0)

        all_preds.append(pred.item())
        all_labels.append(label.item())
        all_probs.append(probs.squeeze(0).cpu().numpy())

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
    print(f"Encryption Time: {running_encryption_time:.2f} seconds")
    print(f"Evaluation Time: {end_eval_time - start_eval_time - running_encryption_time - running_decryption_time:.2f} seconds")
    print(f"Decryption Time: {running_decryption_time:.2f} seconds")
    print(f"Total Time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
