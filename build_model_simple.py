"""
Simple script to build and train the AI image detection model
This script uses TensorFlow CPU to avoid path length issues on Windows
"""

import os
import sys
import argparse

def check_tensorflow():
    """Check if TensorFlow is available"""
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
        return True
    except ImportError as e:
        print(f"TensorFlow not available: {e}")
        return False

def enable_long_paths():
    """Try to enable long path support on Windows"""
    if os.name == 'nt':  # Windows
        try:
            import subprocess
            # Try to enable long path support
            result = subprocess.run([
                'reg', 'add', 'HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem',
                '/v', 'LongPathsEnabled', '/t', 'REG_DWORD', '/d', '1', '/f'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("Long path support enabled. You may need to restart your computer.")
            else:
                print("Could not enable long path support. You may need to do this manually.")
        except Exception as e:
            print(f"Could not enable long path support: {e}")

def install_tensorflow_cpu():
    """Try to install TensorFlow CPU version"""
    try:
        import subprocess
        print("Installing TensorFlow CPU version...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', 'tensorflow-cpu', '--user'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("TensorFlow CPU installed successfully!")
            return True
        else:
            print(f"Failed to install TensorFlow CPU: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error installing TensorFlow CPU: {e}")
        return False

def build_model():
    """Build and train the model"""
    try:
        # Import required modules
        from model import AIImageDetector
        from preprocess import ImagePreprocessor
        import tensorflow as tf
        
        print("Building model...")
        
        # Create detector
        detector = AIImageDetector()
        model = detector.build_hybrid_model()
        detector.compile_model()
        
        print("Model built successfully!")
        print(f"Model parameters: {model.count_params():,}")
        
        # Try to load existing model if available
        if os.path.exists('model.h5'):
            print("Loading existing model...")
            try:
                model = tf.keras.models.load_model('model.h5')
                print("Existing model loaded successfully!")
            except Exception as e:
                print(f"Could not load existing model: {e}")
        
        return model
        
    except Exception as e:
        print(f"Error building model: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Build AI Image Detection Model')
    parser.add_argument('--install', action='store_true', 
                       help='Try to install TensorFlow CPU')
    parser.add_argument('--enable-long-paths', action='store_true',
                       help='Try to enable Windows long path support')
    
    args = parser.parse_args()
    
    if args.enable_long_paths:
        enable_long_paths()
    
    if args.install:
        if not install_tensorflow_cpu():
            print("Failed to install TensorFlow CPU")
            return
    
    # Check TensorFlow
    if not check_tensorflow():
        print("TensorFlow is not available. Try running with --install flag.")
        return
    
    # Build model
    model = build_model()
    if model is not None:
        print("\nModel built successfully!")
        print("You can now train the model using train_model.py")
    else:
        print("\nFailed to build model")

if __name__ == "__main__":
    main()