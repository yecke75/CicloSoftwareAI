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

# Automatically download and load Fashion-MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
complete_train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Numero di classi e campioni per classe
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

print("EDA Reports:")
print("- Train data size: ", len(train_set))
print("- Test data size: ", len(test_set))

# Istogramma del subset bilanciato
subset_labels = complete_train_set.targets.numpy()[indices]
unique, counts = np.unique(subset_labels, return_counts=True)
subset_class_dist = dict(zip(unique, counts))
print("- Class distribution (balanced train set): ", subset_class_dist)

plt.figure(figsize=(10, 4))
sns.barplot(x=list(subset_class_dist.keys()), y=list(subset_class_dist.values()))
plt.title("Class distribution in balanced train set")
plt.xlabel("Digit")
plt.ylabel("Count")
plt.show()

# Show sample images
examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title(f"Label: {example_targets[i].item()}")
    plt.axis('off')
plt.tight_layout()
plt.show()

#criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model on CPU
#start = time.time()
#for epoch in tqdm(range(5)):
    #model.train()
    #for images, labels in train_loader:
        #optimizer.zero_grad()
        #outputs = model(images)
        #loss = criterion(outputs, labels)
        #loss.backward()
        #optimizer.step()
        #running_loss += loss.item()
    #print(f"Epoch [{epoch + 1}/5], Loss: {running_loss/len(train_loader):.4f}")
#print(f"Training time: {time.time() - start:.2f} seconds")