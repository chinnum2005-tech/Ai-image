"""
Setup script for AI-Generated Image Detection System
Handles initial setup and dependency installation
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(cmd, shell=False):
    """Run a command and return success status"""
    try:
        subprocess.run(cmd, shell=shell, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python {version.major}.{version.minor} detected")
    return True


def check_node():
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(['node', '--version'], 
                              capture_output=True, text=True)
        version = result.stdout.strip()
        print(f"✅ Node.js {version} detected")
        return True
    except FileNotFoundError:
        print("❌ Node.js not found. Please install Node.js 16+")
        return False


def create_directories():
    """Create necessary directories"""
    dirs = [
        'dataset/real',
        'dataset/fake',
        'uploads',
        'results',
        'models'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Created necessary directories")


def install_python_deps():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    # Upgrade pip first
    run_command([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
    
    # Install requirements
    if os.path.exists('requirements.txt'):
        success = run_command([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        if success:
            print("✅ Python dependencies installed")
        else:
            print("❌ Failed to install some Python dependencies")
            return False
    else:
        print("❌ requirements.txt not found")
        return False
    
    return True


def install_frontend_deps():
    """Install frontend dependencies"""
    print("\n📦 Installing frontend dependencies...")
    
    frontend_dir = Path('frontend')
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    os.chdir(frontend_dir)
    
    # Install npm packages
    success = run_command(['npm', 'install'])
    
    os.chdir('..')
    
    if success:
        print("✅ Frontend dependencies installed")
    else:
        print("❌ Failed to install frontend dependencies")
        return False
    
    return True


def download_sample_model():
    """Download or create a sample model"""
    print("\n🤖 Checking for model...")
    
    if os.path.exists('model.h5'):
        print("✅ Model file found")
        return True
    
    print("⚠️  No model found. Creating a default model...")
    
    # Create a simple default model
    try:
        from model import AIImageDetector
        detector = AIImageDetector()
        detector.build_cnn_only_model()
        detector.compile_model()
        detector.save_model('model.h5')
        print("✅ Default model created (requires training for accuracy)")
        return True
    except Exception as e:
        print(f"❌ Could not create default model: {e}")
        return False


def check_gpu():
    """Check for GPU availability"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU detected: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"   - {gpu.name}")
        else:
            print("ℹ️  No GPU detected. Will use CPU (slower training)")
    except Exception:
        print("ℹ️  Could not check GPU availability")


def main():
    """Main setup function"""
    print("=" * 60)
    print(" AI-GENERATED IMAGE DETECTION SYSTEM - SETUP ")
    print("=" * 60)
    print()
    
    # Check system requirements
    print("🔍 Checking system requirements...")
    
    if not check_python_version():
        sys.exit(1)
    
    if not check_node():
        print("\n⚠️  Please install Node.js and run setup again")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_python_deps():
        print("\n❌ Setup failed. Please check error messages above.")
        sys.exit(1)
    
    if not install_frontend_deps():
        print("\n❌ Setup failed. Please check error messages above.")
        sys.exit(1)
    
    # Check for model
    download_sample_model()
    
    # Check GPU
    print("\n🖥️  Hardware check...")
    check_gpu()
    
    print("\n" + "=" * 60)
    print(" ✅ SETUP COMPLETED SUCCESSFULLY! ")
    print("=" * 60)
    print()
    print("📝 Next steps:")
    print("1. Add training data:")
    print("   - Real images → dataset/real/")
    print("   - AI-generated images → dataset/fake/")
    print()
    print("2. Train the model:")
    print("   python train_model.py --real_dir dataset/real --fake_dir dataset/fake")
    print()
    print("3. Start the application:")
    print("   Windows: start.bat")
    print("   Or manually:")
    print("   - Backend: python app.py")
    print("   - Frontend: cd frontend && npm run dev")
    print()
    print("🌐 The application will be available at http://localhost:3000")
    print()


if __name__ == "__main__":
    main()
