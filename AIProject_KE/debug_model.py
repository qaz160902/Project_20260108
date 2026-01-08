print("Starting debug script...")
import os
import sys

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import DepthwiseConv2D
    print(f"TensorFlow Version: {tf.__version__}")
except Exception as e:
    print(f"Import Error: {e}")
    sys.exit(1)

class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            print(f"Removing 'groups' argument: {kwargs['groups']}")
            kwargs.pop('groups', None)
        super().__init__(**kwargs)

try:
    model_path = "keras_model.h5"
    if not os.path.exists(model_path):
        print("keras_model.h5 not found")
    else:
        print("Attempting to load model with custom objects...")
        model = load_model(model_path, custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D}, compile=False)
        print("Model loaded successfully!")
except Exception as e:
    print(f"ERROR: {e}")