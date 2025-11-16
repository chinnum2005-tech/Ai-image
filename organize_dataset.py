"""
Dataset Organization Script for Dell Latitude E5470
Optimized for i7-6820HQ CPU, 16GB RAM, Windows 11
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm


def organize_ai_images():
    """Move AI-generated images from archive to dataset/fake"""
    
    archive_dir = Path('archive')
    fake_dir = Path('dataset/fake')
    
    if not archive_dir.exists():
        print("❌ Archive folder not found!")
        return False
    
    # Create fake directory
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all PNG files
    ai_images = list(archive_dir.glob('*.png'))
    
    print(f"\n📦 Found {len(ai_images)} AI-generated images in archive/")
    print(f"Moving to dataset/fake/...")
    
    moved = 0
    for img_path in tqdm(ai_images, desc="Moving images"):
        dest_path = fake_dir / img_path.name
        if not dest_path.exists():
            shutil.copy2(img_path, dest_path)
            moved += 1
    
    print(f"✅ Moved {moved} AI-generated images to dataset/fake/")
    print(f"📁 Keeping originals in archive/ as backup")
    
    return True


def check_dataset_status():
    """Check current dataset status"""
    
    real_dir = Path('dataset/real')
    fake_dir = Path('dataset/fake')
    
    real_count = len(list(real_dir.glob('*.*'))) if real_dir.exists() else 0
    fake_count = len(list(fake_dir.glob('*.*'))) if fake_dir.exists() else 0
    
    print("\n" + "=" * 60)
    print(" CURRENT DATASET STATUS ")
    print("=" * 60)
    print(f"\n✅ AI-Generated Images: {fake_count}")
    print(f"⚠️  Real Images: {real_count}")
    
    if real_count == 0:
        print("\n" + "!" * 60)
        print(" WARNING: NO REAL IMAGES FOUND! ")
        print("!" * 60)
        print("\nYou need REAL photographs to train the model.")
        print("The model learns by comparing real vs AI-generated images.")
        print("\n📸 Where to get real images:")
        print("   1. Your own photos from camera/smartphone")
        print("   2. CelebA dataset (celebrity faces)")
        print("   3. FFHQ dataset (high-quality faces)")
        print("   4. Any authentic photographs")
        print("\n💡 Recommended: At least 1000-2000 real images")
        print("   Place them in: dataset/real/")
    
    print("\n" + "=" * 60)
    
    return real_count, fake_count


def optimize_for_training(fake_count):
    """Suggest optimal training parameters for this system"""
    
    print("\n" + "=" * 60)
    print(" OPTIMIZED TRAINING RECOMMENDATIONS ")
    print(" FOR DELL LATITUDE E5470 (i7-6820HQ, 16GB RAM) ")
    print("=" * 60)
    
    # Calculate optimal settings
    if fake_count < 500:
        subset = fake_count
        epochs = 30
        batch_size = 4
    elif fake_count < 1000:
        subset = 500
        epochs = 40
        batch_size = 8
    else:
        subset = 1000
        epochs = 50
        batch_size = 8
    
    print(f"\n🎯 RECOMMENDED SETTINGS:")
    print(f"   Dataset Size: {subset} real + {subset} AI-generated")
    print(f"   Model Type: CNN-only (lighter than hybrid)")
    print(f"   Batch Size: {batch_size} (memory-friendly)")
    print(f"   Epochs: {epochs}")
    print(f"   Estimated Time: {int((subset * 2 / 100) * epochs * 1.5)} - {int((subset * 2 / 100) * epochs * 2)} minutes")
    
    print(f"\n⚡ CPU OPTIMIZATIONS:")
    print(f"   - Using TensorFlow CPU")
    print(f"   - Reduced batch size for 16GB RAM")
    print(f"   - Image size: 256x256 (good balance)")
    print(f"   - No data augmentation during inference")
    print(f"   - Single-threaded feature extraction")
    
    print(f"\n⚠️  SYSTEM CONSIDERATIONS:")
    print(f"   - Close other applications before training")
    print(f"   - Available RAM: ~14GB usable for training")
    print(f"   - CPU will be at 80-100% during training")
    print(f"   - System may be slow during training")
    
    return subset, epochs, batch_size


def create_balanced_subset(real_count, fake_count, target_size=1000):
    """Create a balanced subset for training"""
    
    if real_count == 0:
        print("\n❌ Cannot create balanced dataset without real images!")
        return False
    
    real_dir = Path('dataset/real')
    fake_dir = Path('dataset/fake')
    
    # Determine subset size
    max_size = min(real_count, fake_count, target_size)
    
    print(f"\n📊 Creating balanced subset of {max_size} images per class...")
    
    # We'll use all available images up to max_size
    # No need to copy, just note the counts
    
    print(f"✅ Balanced dataset ready:")
    print(f"   Real images: {min(real_count, max_size)}")
    print(f"   AI images: {min(fake_count, max_size)}")
    print(f"   Total: {min(real_count, max_size) * 2}")
    
    return True


def main():
    """Main execution"""
    
    print("=" * 60)
    print(" AI IMAGE DETECTION - DATASET ORGANIZER ")
    print(" Optimized for Dell Latitude E5470 ")
    print("=" * 60)
    
    # Step 1: Organize AI images
    print("\n[STEP 1] Organizing AI-generated images...")
    organize_ai_images()
    
    # Step 2: Check dataset status
    print("\n[STEP 2] Checking dataset status...")
    real_count, fake_count = check_dataset_status()
    
    # Step 3: Provide recommendations
    print("\n[STEP 3] Analyzing system capabilities...")
    subset_size, epochs, batch_size = optimize_for_training(fake_count)
    
    # Step 4: Next steps
    print("\n" + "=" * 60)
    print(" NEXT STEPS ")
    print("=" * 60)
    
    if real_count == 0:
        print("\n1. ⚠️  GET REAL IMAGES (CRITICAL!):")
        print("   Download or copy at least 1000 real photos")
        print("   Place them in: dataset/real/")
        print("\n2. After adding real images, run this script again")
        print("\n3. Then train using:")
        print("   python train_optimized.py")
    else:
        print("\n✅ Dataset is ready!")
        print(f"   Real images: {real_count}")
        print(f"   AI images: {fake_count}")
        print("\n🚀 Ready to train:")
        print("   python train_optimized.py")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
