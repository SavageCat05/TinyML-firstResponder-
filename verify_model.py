import sys
import os

# Use a relative path to add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.model.architecture import EmergencyIntentModel

def verify_model_creation():
    """
    Tries to build the model to verify its architecture.
    """
    try:
        print("Attempting to build the model...")
        model_builder = EmergencyIntentModel()
        model = model_builder.build_model()
        if model:
            print("✅ Model built successfully!")
            model.summary()
        else:
            print("❌ Model building failed.")
    except Exception as e:
        print(f"❌ An error occurred during model creation: {e}")

if __name__ == "__main__":
    verify_model_creation()
