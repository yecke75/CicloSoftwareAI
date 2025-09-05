import os
from src.train import train_model, evaluate_model
import torch
from src.train import SimpleCNN

def test_model_training():
    test_model_path = "test_model.pth"

    # Train for 1 epoch for testing purposes
    train_model(1, save_path=test_model_path)
    # Check if the model file is created
    assert os.path.exists(test_model_path), "Model file was not created."
    # Check if the model file is not empty and can be loaded
    assert os.path.getsize(test_model_path) > 0, "Model file is empty."
    # Load the model to ensure it can be loaded without errors
    model = SimpleCNN()
    model.load_state_dict(torch.load(test_model_path))

    # Do a simple inference to check if the model is working
    xin = torch.randn(1, 1, 28, 28) # Example input tensor
    model.eval()
    with torch.no_grad():
        output = model(xin)
    assert output.shape == (1, 10), "Model output shape is incorrect."
    
    # Clean up the model file after the test
    os.remove(test_model_path)
    print("Test passed: model trained and saved successfully.")

def test_model_evaluation():
    # First, train and save a model
    # Assumed the tests are independent, so we train a new model here
    test_model_path = "test_model.pth"
    train_model(1, save_path=test_model_path)
    # Now evaluate the model
    evaluate_model(model_path=test_model_path)
    os.remove(test_model_path)
    os.remove("confusion_matrix.png")
    print("Test passed: model evaluated successfully.")