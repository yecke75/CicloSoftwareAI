import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

def run_inference(model_path: str, image_path: str):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Define the same model architecture as used during training
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

    # Load the model
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))

    # Preprocess the image
    try:
        image = Image.open(image_path).convert('L')  # Open in grayscale
    except FileNotFoundError:
        raise ValueError(f"Image not found or unable to open: {image_path}")
    
     # Resize the image to 28x28 and convert to numpy array
    image = image.resize((28, 28), Image.LANCZOS)
    image = np.array(image).astype('float32') / 255.0
    
    # Convert numpy array to torch tensor and add dimensions
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 28, 28)

    # Run inference
    print(f"Running inference on image: {image_path} using model {model_path}...")
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
    
    predicted_class = class_names[predicted.item()]
    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a Fashion-MNIST image using a trained model.")
    parser.add_argument("model", type=str, help="Path to the trained model file (e.g., model.pth).", default="app/results/model.pth")
    parser.add_argument("image", type=str, help="Path to the input image file (e.g., image.png).", default="app/results/sample_image.png")
    args = parser.parse_args()

    run_inference(args.model, args.image)