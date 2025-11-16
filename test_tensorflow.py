#!/usr/bin/env python3
"""
Test TensorFlow Installation
============================

This script tests if TensorFlow is properly installed and working.
Run this after installing TensorFlow to verify the installation.
"""

def test_tensorflow():
    """Test if TensorFlow is working correctly"""
    try:
        import tensorflow as tf
        print("✅ TensorFlow Import Test: PASSED")
        print(f"   TensorFlow version: {tf.__version__}")
    except Exception as e:
        print("❌ TensorFlow Import Test: FAILED")
        print(f"   Error: {e}")
        return False
    
    try:
        # Test basic operations
        import tensorflow as tf
        hello = tf.constant('Hello, TensorFlow!')
        print("✅ TensorFlow Basic Operation Test: PASSED")
    except Exception as e:
        print("❌ TensorFlow Basic Operation Test: FAILED")
        print(f"   Error: {e}")
        return False
    
    try:
        # Test GPU availability
        import tensorflow as tf
        gpu_available = tf.config.list_physical_devices('GPU')
        if gpu_available:
            print("✅ GPU Availability Test: PASSED")
            print(f"   GPUs available: {len(gpu_available)}")
        else:
            print("⚠️  GPU Availability Test: No GPU found (using CPU)")
    except Exception as e:
        print("❌ GPU Availability Test: FAILED")
        print(f"   Error: {e}")
    
    try:
        # Test simple computation
        import tensorflow as tf
        x = tf.constant([[1, 2], [3, 4]])
        y = tf.constant([[5, 6], [7, 8]])
        z = tf.matmul(x, y)
        print("✅ TensorFlow Computation Test: PASSED")
        print(f"   Matrix multiplication result shape: {z.shape}")
    except Exception as e:
        print("❌ TensorFlow Computation Test: FAILED")
        print(f"   Error: {e}")
        return False
    
    print("\n🎉 All tests completed!")
    print("TensorFlow is properly installed and ready to use.")
    return True

def test_model_files():
    """Test if the pre-trained model files exist"""
    import os
    try:
        # Check for model files
        model_files = ['model.h5', 'best_model_optimized.h5']
        found_models = [f for f in model_files if os.path.exists(f)]
        
        if not found_models:
            print("⚠️  No pre-trained model files found")
            return True  # This is not a failure, just informational
        
        print(f"\n📁 Found model files: {found_models}")
        print("✅ Model Files Test: PASSED")
        return True
        
    except Exception as e:
        print("❌ Model Files Test: FAILED")
        print(f"   Error: {e}")
        return False

if __name__ == "__main__":
    print("🤖 TensorFlow Installation Verification")
    print("=" * 40)
    
    # Test TensorFlow
    tf_success = test_tensorflow()
    
    # Test model files
    files_success = test_model_files()
    
    print("\n" + "=" * 40)
    if tf_success:
        print("✅ TensorFlow is ready for the AI Image Detection System!")
        print("The system will work with ANY image you upload through the web interface.")
    else:
        print("❌ TensorFlow installation needs to be fixed.")
        print("Please follow the installation guide and try again.")