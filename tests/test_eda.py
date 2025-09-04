import os
from src.eda import eda
import warnings

def test_eda():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    eda() # Run EDA function to ensure it executes without errors
    os.remove("class_distribution.png")
    os.remove("sample_images.png")
    print("EDA test passed: EDA function ecxecuted successfully.")
