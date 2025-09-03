import os
from src.eda import eda

def test_eda():
    eda() # Run EDA function to ensure it executes without errors
    os.remove("class_distribution.png")
    os.remove("sample_images.png")
    print("EDA test passed: EDA function ecxecuted successfully.")
