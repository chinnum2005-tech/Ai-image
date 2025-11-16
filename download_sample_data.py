"""
Download Sample Dataset for AI-Generated Image Detection
Downloads sample images for testing and demonstration
"""

import os
import urllib.request
import zipfile
from pathlib import Path
import json
import hashlib


def download_file(url, destination):
    """Download a file from URL to destination"""
    try:
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, destination)
        return True
    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def create_sample_dataset():
    """Create a small sample dataset for testing"""
    import numpy as np
    from PIL import Image
    
    # Create directories
    Path('dataset/real').mkdir(parents=True, exist_ok=True)
    Path('dataset/fake').mkdir(parents=True, exist_ok=True)
    
    print("Creating sample dataset...")
    
    # Generate sample "real" images (with noise patterns)
    for i in range(10):
        # Create image with camera-like noise
        img_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        # Add some structure (not just noise)
        x, y = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
        pattern = (np.sin(10 * x) * np.cos(10 * y) * 127 + 128).astype(np.uint8)
        img_array[:, :, 0] = (img_array[:, :, 0] * 0.3 + pattern * 0.7).astype(np.uint8)
        img_array[:, :, 1] = (img_array[:, :, 1] * 0.3 + pattern * 0.7).astype(np.uint8)
        img_array[:, :, 2] = (img_array[:, :, 2] * 0.3 + np.roll(pattern, 50) * 0.7).astype(np.uint8)
        
        # Add camera-like noise
        noise = np.random.normal(0, 5, (256, 256, 3))
        img_array = np.clip(img_array.astype(float) + noise, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(img_array)
        img.save(f'dataset/real/sample_real_{i+1:03d}.jpg', quality=95)
    
    # Generate sample "AI" images (smoother, less noise)
    for i in range(10):
        # Create smoother image (AI-like)
        img_array = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Create smooth gradients
        x, y = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
        
        # Smooth color patterns
        img_array[:, :, 0] = (np.sin(5 * x) * 127 + 128).astype(np.uint8)
        img_array[:, :, 1] = (np.cos(5 * y) * 127 + 128).astype(np.uint8)
        img_array[:, :, 2] = (np.sin(3 * x) * np.cos(3 * y) * 127 + 128).astype(np.uint8)
        
        # Apply Gaussian blur for smoothness (AI characteristic)
        from scipy.ndimage import gaussian_filter
        img_array = gaussian_filter(img_array, sigma=2)
        
        # Very minimal noise (AI characteristic)
        noise = np.random.normal(0, 1, (256, 256, 3))
        img_array = np.clip(img_array.astype(float) + noise, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(img_array)
        img.save(f'dataset/fake/sample_ai_{i+1:03d}.jpg', quality=98)
    
    print("✅ Created 10 sample real images")
    print("✅ Created 10 sample AI-generated images")
    return True


def download_public_dataset():
    """
    Information about public datasets (URLs not included for safety)
    """
    print("\n" + "=" * 60)
    print(" PUBLIC DATASET SOURCES ")
    print("=" * 60)
    print()
    print("Recommended public datasets for training:")
    print()
    print("1. Real Face Images:")
    print("   - CelebA Dataset")
    print("   - FFHQ (Flickr-Faces-HQ)")
    print("   - LFW (Labeled Faces in the Wild)")
    print()
    print("2. AI-Generated Images:")
    print("   - StyleGAN Generated Faces")
    print("   - ThisPersonDoesNotExist samples")
    print("   - 100K Faces (Generated)")
    print()
    print("3. Mixed Datasets:")
    print("   - DFDC (Deepfake Detection Challenge)")
    print("   - FaceForensics++")
    print()
    print("Please download these datasets from their official sources")
    print("and place them in the appropriate directories:")
    print("  - dataset/real/ for authentic images")
    print("  - dataset/fake/ for AI-generated images")
    print()


def main():
    print("=" * 60)
    print(" DATASET PREPARATION ")
    print("=" * 60)
    print()
    
    # Check existing dataset
    real_dir = Path('dataset/real')
    fake_dir = Path('dataset/fake')
    
    existing_real = len(list(real_dir.glob('*.*'))) if real_dir.exists() else 0
    existing_fake = len(list(fake_dir.glob('*.*'))) if fake_dir.exists() else 0
    
    if existing_real > 0 or existing_fake > 0:
        print(f"Existing dataset found:")
        print(f"  Real images: {existing_real}")
        print(f"  Fake images: {existing_fake}")
        print()
        
        response = input("Do you want to keep existing files? (y/n): ")
        if response.lower() == 'n':
            print("Clearing existing dataset...")
            import shutil
            if real_dir.exists():
                shutil.rmtree(real_dir)
            if fake_dir.exists():
                shutil.rmtree(fake_dir)
            real_dir.mkdir(parents=True, exist_ok=True)
            fake_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nOptions:")
    print("1. Create small sample dataset (for testing)")
    print("2. Get information about public datasets")
    print("3. Exit")
    
    while True:
        choice = input("\nSelect option (1-3): ")
        if choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    if choice == '1':
        try:
            create_sample_dataset()
            print("\n✅ Sample dataset created successfully!")
            print("\nDataset location:")
            print("  - dataset/real/ (10 sample real-like images)")
            print("  - dataset/fake/ (10 sample AI-like images)")
            print("\n⚠️  Note: This is a minimal dataset for testing only.")
            print("   For accurate detection, use real photographs and")
            print("   actual AI-generated images (minimum 100 each).")
        except Exception as e:
            print(f"\n❌ Error creating sample dataset: {e}")
    
    elif choice == '2':
        download_public_dataset()
    
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()
