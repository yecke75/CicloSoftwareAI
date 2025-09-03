import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Subset

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
def train_model(epochs: int, save_path: str = "model.pth"):
    # Automatically download and load Fashion-MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    complete_train_set = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    num_classes = 10
    samples_per_class = 2000

    # Ottieni gli indici per ciascuna classe
    indices = []
    labels = complete_train_set.targets.numpy()
    for cls in range(num_classes):
        cls_indices = np.where(labels == cls)[0][:samples_per_class]
        indices.extend(cls_indices)

    # Crea il subset bilanciato
    train_set = Subset(complete_train_set, indices)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model on CPU
    start = time.time()
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
        print(f"Epoch [{epoch + 1}], Loss: {running_loss/len(train_loader):.4f}")
    print(f"Training time: {time.time() - start:.2f} seconds")

    torch.save(model.state_dict(), save_path)

def evaluate_model(model_path: str = "model.pth"):
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    accuracy = correct / total
    print(f"\nTest accuracy: {accuracy:.4f}")

    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))

    # Plot confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show(block=False)
    plt.savefig("confusion_matrix.png")
