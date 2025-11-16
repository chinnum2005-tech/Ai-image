"""
Quick Training Script for AI-Generated Image Detection
Provides an interactive interface for model training
"""

import os
import sys
from pathlib import Path
import argparse


def check_dataset():
    """Check if dataset directories exist and contain images"""
    real_dir = Path('dataset/real')
    fake_dir = Path('dataset/fake')
    
    real_images = []
    fake_images = []
    
    if real_dir.exists():
        real_images = list(real_dir.glob('*.jpg')) + list(real_dir.glob('*.png')) + \
                     list(real_dir.glob('*.jpeg')) + list(real_dir.glob('*.bmp'))
    
    if fake_dir.exists():
        fake_images = list(fake_dir.glob('*.jpg')) + list(fake_dir.glob('*.png')) + \
                     list(fake_dir.glob('*.jpeg')) + list(fake_dir.glob('*.bmp'))
    
    return len(real_images), len(fake_images)


def main():
    print("=" * 60)
    print(" AI IMAGE DETECTION - QUICK TRAINING ")
    print("=" * 60)
    print()
    
    # Check dataset
    real_count, fake_count = check_dataset()
    
    if real_count == 0 or fake_count == 0:
        print("❌ Dataset not found!")
        print()
        print("Please add images to:")
        print("  - dataset/real/ (for real photographs)")
        print("  - dataset/fake/ (for AI-generated images)")
        print()
        print("You need at least 50 images in each category for basic training.")
        sys.exit(1)
    
    print(f"✅ Dataset found:")
    print(f"   Real images: {real_count}")
    print(f"   AI-generated: {fake_count}")
    print()
    
    if real_count < 50 or fake_count < 50:
        print("⚠️  Warning: You have less than 50 images in some categories.")
        print("   This may result in poor model performance.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Select model type
    print("\nSelect model type:")
    print("1. Hybrid (CNN + Pixel Features) - Most accurate, slower")
    print("2. CNN Only - Good accuracy, moderate speed")
    print("3. Lightweight - Fastest, lower accuracy")
    
    while True:
        choice = input("\nEnter choice (1-3): ")
        if choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    model_types = {
        '1': 'hybrid',
        '2': 'cnn',
        '3': 'lightweight'
    }
    model_type = model_types[choice]
    
    # Select training duration
    print("\nSelect training duration:")
    print("1. Quick (10 epochs) - ~5-10 minutes")
    print("2. Standard (30 epochs) - ~15-30 minutes")
    print("3. Extended (50 epochs) - ~30-60 minutes")
    print("4. Custom")
    
    while True:
        duration_choice = input("\nEnter choice (1-4): ")
        if duration_choice in ['1', '2', '3', '4']:
            break
        print("Invalid choice. Please enter 1, 2, 3, or 4.")
    
    epochs_map = {
        '1': 10,
        '2': 30,
        '3': 50
    }
    
    if duration_choice == '4':
        while True:
            try:
                epochs = int(input("Enter number of epochs (1-200): "))
                if 1 <= epochs <= 200:
                    break
                print("Please enter a number between 1 and 200.")
            except ValueError:
                print("Please enter a valid number.")
    else:
        epochs = epochs_map[duration_choice]
    
    # Batch size
    print("\nSelect batch size:")
    print("1. Small (8) - Less memory, slower")
    print("2. Medium (16) - Balanced")
    print("3. Large (32) - More memory, faster")
    
    while True:
        batch_choice = input("\nEnter choice (1-3): ")
        if batch_choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    batch_sizes = {
        '1': 8,
        '2': 16,
        '3': 32
    }
    batch_size = batch_sizes[batch_choice]
    
    # Data augmentation
    augment_response = input("\nEnable data augmentation? (recommended) (y/n): ")
    use_augmentation = augment_response.lower() != 'n'
    
    # Summary
    print("\n" + "=" * 60)
    print(" TRAINING CONFIGURATION ")
    print("=" * 60)
    print(f"Model Type: {model_type.upper()}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Data Augmentation: {'Enabled' if use_augmentation else 'Disabled'}")
    print(f"Dataset: {real_count} real, {fake_count} fake images")
    print("=" * 60)
    
    confirm = input("\nStart training? (y/n): ")
    if confirm.lower() != 'y':
        print("Training cancelled.")
        sys.exit(0)
    
    # Build command
    cmd = [
        sys.executable,
        'train_model.py',
        '--real_dir', 'dataset/real',
        '--fake_dir', 'dataset/fake',
        '--model_type', model_type,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size)
    ]
    
    if not use_augmentation:
        cmd.append('--no_augmentation')
    
    print("\n🚀 Starting training...")
    print("=" * 60)
    
    # Run training
    import subprocess
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
        print("Model saved to: model.h5")
        print("\nYou can now run the application using:")
        print("  Windows: start.bat")
        print("  Or manually start the backend and frontend")
    except subprocess.CalledProcessError:
        print("\n❌ Training failed. Please check the error messages above.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
