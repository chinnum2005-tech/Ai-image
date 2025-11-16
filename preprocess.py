"""
Image Preprocessing Module for AI-Generated Image Detection
Handles normalization, resizing, and data augmentation
"""

import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import random


class ImagePreprocessor:
    """
    Preprocesses images for AI detection model
    """
    
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
        
    def preprocess_image(self, image_path, augment=False):
        """
        Preprocess a single image
        """
        # Read image
        img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation if training
        if augment:
            img = self.augment_image(img)
        
        # Resize while maintaining aspect ratio
        img = self.resize_with_padding(img)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def resize_with_padding(self, img):
        """
        Resize image to target size while maintaining aspect ratio
        Pad with black if necessary
        """
        h, w = img.shape[:2]
        target_h, target_w = self.target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calculate padding
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        
        # Place resized image in center
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        return padded
    
    def augment_image(self, img):
        """
        Apply data augmentation techniques
        """
        # Random horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        
        # Random rotation (-10 to 10 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            h, w = img.shape[:2]
            matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            img = cv2.warpAffine(img, matrix, (w, h))
        
        # Random brightness adjustment
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
        
        # Random contrast adjustment
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            img = cv2.convertScaleAbs(img, alpha=factor, beta=128*(1-factor))
        
        # Random noise
        if random.random() > 0.3:
            noise = np.random.randn(*img.shape) * random.uniform(5, 15)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def preprocess_batch(self, image_paths, augment=False):
        """
        Preprocess a batch of images
        """
        images = []
        valid_paths = []
        
        for path in image_paths:
            try:
                img = self.preprocess_image(path, augment)
                images.append(img)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        if images:
            return np.array(images), valid_paths
        else:
            return np.array([]), []
    
    def normalize_features(self, features):
        """
        Normalize feature vector to zero mean and unit variance
        """
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        
        # Avoid division by zero
        std[std == 0] = 1
        
        normalized = (features - mean) / std
        
        return normalized, mean, std
    
    def apply_normalization(self, features, mean, std):
        """
        Apply pre-computed normalization parameters
        """
        std[std == 0] = 1
        return (features - mean) / std
    
    def prepare_for_cnn(self, image_path):
        """
        Prepare image for CNN input
        """
        img = self.preprocess_image(image_path, augment=False)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def load_and_split_dataset(self, real_dir, fake_dir, test_split=0.2, val_split=0.1):
        """
        Load dataset and split into train/val/test sets
        """
        # Get all image paths
        real_images = []
        fake_images = []
        
        # Supported image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # Load real images
        if os.path.exists(real_dir):
            for ext in extensions:
                real_images.extend(Path(real_dir).glob(f'*{ext}'))
                real_images.extend(Path(real_dir).glob(f'*{ext.upper()}'))
        
        # Load fake images
        if os.path.exists(fake_dir):
            for ext in extensions:
                fake_images.extend(Path(fake_dir).glob(f'*{ext}'))
                fake_images.extend(Path(fake_dir).glob(f'*{ext.upper()}'))
        
        print(f"Found {len(real_images)} real images and {len(fake_images)} fake images")
        
        # Create labels (0 for real, 1 for fake)
        real_labels = [0] * len(real_images)
        fake_labels = [1] * len(fake_images)
        
        # Combine and shuffle
        all_images = real_images + fake_images
        all_labels = real_labels + fake_labels
        
        # Shuffle
        combined = list(zip(all_images, all_labels))
        random.shuffle(combined)
        all_images, all_labels = zip(*combined) if combined else ([], [])
        
        # Calculate split indices
        total = len(all_images)
        test_size = int(total * test_split)
        val_size = int(total * val_split)
        train_size = total - test_size - val_size
        
        # Split dataset
        train_images = list(all_images[:train_size])
        train_labels = list(all_labels[:train_size])
        
        val_images = list(all_images[train_size:train_size+val_size])
        val_labels = list(all_labels[train_size:train_size+val_size])
        
        test_images = list(all_images[train_size+val_size:])
        test_labels = list(all_labels[train_size+val_size:])
        
        return {
            'train': (train_images, train_labels),
            'val': (val_images, val_labels),
            'test': (test_images, test_labels)
        }
    
    def create_augmented_dataset(self, images, labels, augmentation_factor=2):
        """
        Create augmented dataset for training
        """
        augmented_images = []
        augmented_labels = []
        
        for img_path, label in zip(images, labels):
            # Add original
            augmented_images.append(img_path)
            augmented_labels.append(label)
            
            # Add augmented versions
            for _ in range(augmentation_factor - 1):
                augmented_images.append(img_path)
                augmented_labels.append(label)
        
        return augmented_images, augmented_labels


class DataGenerator:
    """
    Data generator for training with batch loading
    """
    
    def __init__(self, image_paths, labels, batch_size=32, preprocessor=None, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.augment = augment
        self.indices = np.arange(len(image_paths))
        
    def __len__(self):
        return len(self.image_paths) // self.batch_size
    
    def __iter__(self):
        # Shuffle indices
        np.random.shuffle(self.indices)
        
        for i in range(0, len(self.indices) - self.batch_size, self.batch_size):
            batch_indices = self.indices[i:i+self.batch_size]
            batch_paths = [self.image_paths[j] for j in batch_indices]
            batch_labels = [self.labels[j] for j in batch_indices]
            
            # Preprocess batch
            images, valid_paths = self.preprocessor.preprocess_batch(batch_paths, self.augment)
            
            if len(images) > 0:
                # Get corresponding labels for valid images
                valid_labels = []
                for path, label in zip(batch_paths, batch_labels):
                    if path in valid_paths:
                        valid_labels.append(label)
                
                yield np.array(images), np.array(valid_labels)
    
    def get_batch(self, batch_index):
        """
        Get a specific batch
        """
        start = batch_index * self.batch_size
        end = min(start + self.batch_size, len(self.image_paths))
        
        batch_paths = self.image_paths[start:end]
        batch_labels = self.labels[start:end]
        
        images, _ = self.preprocessor.preprocess_batch(batch_paths, self.augment)
        
        return images, np.array(batch_labels)


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = ImagePreprocessor()
    print("Image Preprocessor initialized successfully!")
    print(f"Target size: {preprocessor.target_size}")
    
    # Test data generator
    gen = DataGenerator([], [], batch_size=32)
    print(f"Data Generator created with batch size: {gen.batch_size}")
