"""
Final Dataset Preparation Script
Organizes your real images (PIC/) and AI images (archive/) for training
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm


def convert_heic_to_jpg(heic_path, jpg_path):
    """Convert HEIC to JPG if possible"""
    try:
        from PIL import Image
        import pillow_heif
        
        # Register HEIF opener
        pillow_heif.register_heif_opener()
        
        # Open and convert
        img = Image.open(heic_path)
        img = img.convert('RGB')
        img.save(jpg_path, 'JPEG', quality=95)
        return True
    except ImportError:
        print(f"⚠️  Cannot convert HEIC files (pillow-heif not installed)")
        print(f"   Installing: pip install pillow-heif")
        return False
    except Exception as e:
        print(f"⚠️  Error converting {heic_path.name}: {e}")
        return False


def organize_real_images():
    """Organize real images from PIC/ to dataset/real/"""
    
    pic_dir = Path('PIC')
    real_dir = Path('dataset/real')
    
    if not pic_dir.exists():
        print("❌ PIC folder not found!")
        return 0
    
    # Create real directory
    real_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.heic', '.HEIC']
    all_images = []
    for ext in image_extensions:
        all_images.extend(list(pic_dir.glob(f'*{ext}')))
    
    print(f"\n📸 Found {len(all_images)} real images in PIC/")
    print(f"Organizing to dataset/real/...")
    
    copied = 0
    converted = 0
    skipped = 0
    
    for img_path in tqdm(all_images, desc="Processing images"):
        # Determine destination
        if img_path.suffix.lower() in ['.heic']:
            # Convert HEIC to JPG
            dest_path = real_dir / f"{img_path.stem}.jpg"
            if not dest_path.exists():
                if convert_heic_to_jpg(img_path, dest_path):
                    converted += 1
                else:
                    skipped += 1
        else:
            # Copy JPG/PNG directly
            dest_path = real_dir / img_path.name
            if not dest_path.exists():
                shutil.copy2(img_path, dest_path)
                copied += 1
    
    print(f"\n✅ Organized {copied} images")
    if converted > 0:
        print(f"✅ Converted {converted} HEIC files to JPG")
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} HEIC files (install pillow-heif)")
    print(f"📁 Keeping originals in PIC/ as backup")
    
    return copied + converted


def organize_ai_images():
    """Organize AI images from archive/ to dataset/fake/"""
    
    archive_dir = Path('archive')
    fake_dir = Path('dataset/fake')
    
    if not archive_dir.exists():
        print("❌ Archive folder not found!")
        return 0
    
    # Create fake directory
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all PNG files
    ai_images = list(archive_dir.glob('*.png'))
    
    if len(ai_images) == 0:
        print("⚠️  No AI images found in archive/")
        return 0
    
    print(f"\n🤖 Found {len(ai_images)} AI-generated images in archive/")
    print(f"Organizing to dataset/fake/...")
    
    moved = 0
    for img_path in tqdm(ai_images, desc="Processing images"):
        dest_path = fake_dir / img_path.name
        if not dest_path.exists():
            shutil.copy2(img_path, dest_path)
            moved += 1
    
    print(f"✅ Organized {moved} AI-generated images")
    print(f"📁 Keeping originals in archive/ as backup")
    
    return moved


def check_final_status():
    """Check final dataset status"""
    
    real_dir = Path('dataset/real')
    fake_dir = Path('dataset/fake')
    
    real_count = len(list(real_dir.glob('*.*'))) if real_dir.exists() else 0
    fake_count = len(list(fake_dir.glob('*.*'))) if fake_dir.exists() else 0
    
    print("\n" + "=" * 60)
    print(" FINAL DATASET STATUS ")
    print("=" * 60)
    print(f"\n✅ Real Images: {real_count}")
    print(f"✅ AI-Generated Images: {fake_count}")
    print(f"📊 Total Images: {real_count + fake_count}")
    
    # Determine training recommendation
    min_count = min(real_count, fake_count)
    
    if min_count < 100:
        print(f"\n⚠️  WARNING: Only {min_count} images per class")
        print(f"   Recommended: At least 500 per class")
        print(f"   Training will work but accuracy may be low (<75%)")
    elif min_count < 500:
        print(f"\n⚠️  Low dataset size: {min_count} images per class")
        print(f"   Recommended: 500-1000 per class for better accuracy")
        print(f"   Expected accuracy: 70-85%")
    else:
        print(f"\n✅ Good dataset size: {min_count} images per class")
        print(f"   Expected accuracy: 85-95%")
    
    print("\n" + "=" * 60)
    
    return real_count, fake_count


def generate_training_commands(real_count, fake_count):
    """Generate training commands based on dataset"""
    
    min_count = min(real_count, fake_count)
    
    print("\n" + "=" * 60)
    print(" READY TO TRAIN! ")
    print("=" * 60)
    
    print("\n📋 MANUAL TRAINING COMMANDS:")
    print("\n" + "-" * 60)
    
    if min_count >= 1000:
        print("🎯 RECOMMENDED (Full Dataset):")
        print("python train_optimized.py --max_images 1000 --epochs 40 --batch_size 8")
        print(f"   Time: ~60-90 minutes")
        print(f"   Expected accuracy: 90-95%")
        
        print("\n⚡ QUICK TEST (Subset):")
        print("python train_optimized.py --max_images 500 --epochs 20 --batch_size 8")
        print(f"   Time: ~25-35 minutes")
        print(f"   Expected accuracy: 80-88%")
        
    elif min_count >= 500:
        print("🎯 RECOMMENDED (Medium Dataset):")
        print(f"python train_optimized.py --max_images {min_count} --epochs 30 --batch_size 8")
        print(f"   Time: ~40-60 minutes")
        print(f"   Expected accuracy: 85-92%")
        
        print("\n⚡ QUICK TEST:")
        print(f"python train_optimized.py --max_images {min_count//2} --epochs 20 --batch_size 8")
        print(f"   Time: ~20-30 minutes")
        print(f"   Expected accuracy: 75-85%")
        
    else:
        print("🎯 RECOMMENDED (Small Dataset):")
        print(f"python train_optimized.py --max_images {min_count} --epochs 25 --batch_size 4")
        print(f"   Time: ~15-25 minutes")
        print(f"   Expected accuracy: 70-80%")
    
    print("\n" + "-" * 60)
    
    print("\n💡 TIPS:")
    print("   • Close other applications before training")
    print("   • Training will use 80-100% CPU (normal)")
    print("   • Your system may be slow during training")
    print("   • Don't interrupt the process")
    
    print("\n📊 AFTER TRAINING:")
    print("   • Model will be saved as model.h5")
    print("   • Test with: python app.py")
    print("   • Launch frontend: cd frontend && npm run dev")
    
    print("\n" + "=" * 60)


def main():
    """Main execution"""
    
    print("=" * 60)
    print(" FINAL PREPARATION FOR TRAINING ")
    print(" Dell Latitude E5470 - Optimized Setup ")
    print("=" * 60)
    
    # Step 1: Organize real images
    print("\n[STEP 1/3] Organizing real images from PIC/...")
    real_organized = organize_real_images()
    
    # Step 2: Organize AI images
    print("\n[STEP 2/3] Organizing AI images from archive/...")
    fake_organized = organize_ai_images()
    
    # Step 3: Check final status
    print("\n[STEP 3/3] Verifying dataset...")
    real_count, fake_count = check_final_status()
    
    # Generate training commands
    if real_count > 0 and fake_count > 0:
        generate_training_commands(real_count, fake_count)
        
        print("\n✅ ALL SET! Dataset is ready for training.")
        print("\n🚀 NEXT STEP: Copy and run one of the commands above!")
    else:
        print("\n❌ Dataset preparation incomplete!")
        if real_count == 0:
            print("   Missing: Real images in dataset/real/")
        if fake_count == 0:
            print("   Missing: AI images in dataset/fake/")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
