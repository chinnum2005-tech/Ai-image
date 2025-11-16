"""
Optimized Training Script for Dell Latitude E5470
i7-6820HQ CPU, 16GB RAM, Windows 11 - CPU ONLY
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optimize for CPU

import numpy as np
import tensorflow as tf
from pathlib import Path
import time
from datetime import datetime
import psutil
import gc

# Force CPU usage (no GPU)
tf.config.set_visible_devices([], 'GPU')

from preprocess import ImagePreprocessor
from model import AIImageDetector
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class OptimizedTrainer:
    """
    Optimized trainer for CPU-only system with 16GB RAM
    """
    
    def __init__(self, real_dir='dataset/real', fake_dir='dataset/fake'):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.preprocessor = ImagePreprocessor(target_size=(256, 256))
        self.detector = None
        self.history = None
        
        # Check system resources
        self.check_system_resources()
    
    def check_system_resources(self):
        """Check available system resources"""
        mem = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        print("\n" + "=" * 60)
        print(" SYSTEM RESOURCES ")
        print("=" * 60)
        print(f"CPU Cores: {cpu_count}")
        print(f"Total RAM: {mem.total / (1024**3):.1f} GB")
        print(f"Available RAM: {mem.available / (1024**3):.1f} GB")
        print(f"RAM Usage: {mem.percent}%")
        
        if mem.available < 4 * 1024**3:  # Less than 4GB available
            print("\n⚠️  WARNING: Low available RAM!")
            print("   Close other applications before training")
            print("   Press Ctrl+C to cancel, or wait 5 seconds to continue...")
            time.sleep(5)
    
    def prepare_dataset(self, max_images_per_class=1000):
        """
        Prepare dataset with memory optimization
        """
        print("\n" + "=" * 60)
        print(" PREPARING DATASET ")
        print("=" * 60)
        
        # Get image paths
        real_images = list(Path(self.real_dir).glob('*.*'))[:max_images_per_class]
        fake_images = list(Path(self.fake_dir).glob('*.*'))[:max_images_per_class]
        
        print(f"\nReal images: {len(real_images)}")
        print(f"AI-generated images: {len(fake_images)}")
        
        if len(real_images) == 0:
            raise ValueError("No real images found! Add photos to dataset/real/")
        
        if len(fake_images) == 0:
            raise ValueError("No AI images found! Run organize_dataset.py first")
        
        # Balance dataset
        min_count = min(len(real_images), len(fake_images))
        real_images = real_images[:min_count]
        fake_images = fake_images[:min_count]
        
        print(f"\n✅ Balanced dataset: {min_count} images per class")
        print(f"Total images: {min_count * 2}")
        
        # Create labels
        all_images = real_images + fake_images
        all_labels = [0] * len(real_images) + [1] * len(fake_images)
        
        # Shuffle
        indices = np.random.permutation(len(all_images))
        all_images = [all_images[i] for i in indices]
        all_labels = [all_labels[i] for i in indices]
        
        # Split dataset
        train_split = int(len(all_images) * 0.7)
        val_split = int(len(all_images) * 0.85)
        
        self.train_paths = all_images[:train_split]
        self.train_labels = all_labels[:train_split]
        
        self.val_paths = all_images[train_split:val_split]
        self.val_labels = all_labels[train_split:val_split]
        
        self.test_paths = all_images[val_split:]
        self.test_labels = all_labels[val_split:]
        
        print(f"\nDataset split:")
        print(f"  Training: {len(self.train_paths)} images")
        print(f"  Validation: {len(self.val_paths)} images")
        print(f"  Test: {len(self.test_paths)} images")
        
        return True
    
    def create_data_generator(self, paths, labels, batch_size, augment=False):
        """
        Memory-efficient data generator
        """
        def generator():
            indices = np.arange(len(paths))
            if augment:
                np.random.shuffle(indices)
            
            for i in range(0, len(paths) - batch_size, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_images = []
                batch_labels = []
                
                for idx in batch_indices:
                    try:
                        img = self.preprocessor.preprocess_image(
                            str(paths[idx]), 
                            augment=augment
                        )
                        batch_images.append(img)
                        batch_labels.append(labels[idx])
                    except Exception as e:
                        continue
                
                if batch_images:
                    yield np.array(batch_images), np.array(batch_labels)
        
        return generator
    
    def build_model(self):
        """Build optimized CNN model"""
        print("\n" + "=" * 60)
        print(" BUILDING MODEL ")
        print("=" * 60)
        
        self.detector = AIImageDetector(input_shape=(256, 256, 3))
        
        # Build CNN-only model (lighter than hybrid)
        self.detector.build_cnn_only_model()
        
        # Compile with optimized settings for CPU
        self.detector.compile_model(learning_rate=0.001)
        
        print(f"\n✅ Model built successfully")
        print(f"Total parameters: {self.detector.model.count_params():,}")
        print(f"Model type: CNN-only (optimized for CPU)")
    
    def train(self, epochs=30, batch_size=8):
        """Train the model with memory optimization"""
        print("\n" + "=" * 60)
        print(" TRAINING MODEL ")
        print("=" * 60)
        print(f"\nEpochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Estimated time: {int((len(self.train_paths) / 100) * epochs * 1.5)}-{int((len(self.train_paths) / 100) * epochs * 2)} minutes")
        
        # Calculate steps
        steps_per_epoch = len(self.train_paths) // batch_size
        validation_steps = len(self.val_paths) // batch_size
        
        # Create generators
        train_gen = tf.data.Dataset.from_generator(
            self.create_data_generator(self.train_paths, self.train_labels, batch_size, augment=True),
            output_signature=(
                tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        ).prefetch(tf.data.AUTOTUNE)
        
        val_gen = tf.data.Dataset.from_generator(
            self.create_data_generator(self.val_paths, self.val_labels, batch_size, augment=False),
            output_signature=(
                tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        ).prefetch(tf.data.AUTOTUNE)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                'best_model_optimized.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: [
                    gc.collect(),  # Force garbage collection
                    print(f"\nRAM: {psutil.virtual_memory().percent}% | "
                          f"Available: {psutil.virtual_memory().available / (1024**3):.1f}GB")
                ]
            )
        ]
        
        # Train
        start_time = time.time()
        
        try:
            self.history = self.detector.model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_gen,
                validation_steps=validation_steps,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        except KeyboardInterrupt:
            print("\n\n⚠️ Training interrupted by user")
            return False
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time/60:.1f} minutes")
        
        # Save final model
        self.detector.save_model('model.h5')
        
        return True
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("\n" + "=" * 60)
        print(" EVALUATING MODEL ")
        print("=" * 60)
        
        # Prepare test data
        test_images = []
        valid_labels = []
        
        print("\nLoading test images...")
        for path, label in zip(self.test_paths, self.test_labels):
            try:
                img = self.preprocessor.preprocess_image(str(path), augment=False)
                test_images.append(img)
                valid_labels.append(label)
            except:
                continue
        
        test_images = np.array(test_images)
        valid_labels = np.array(valid_labels)
        
        print(f"Test set: {len(test_images)} images")
        
        # Predict
        print("\nMaking predictions...")
        predictions = self.detector.model.predict(test_images, batch_size=8, verbose=1)
        binary_preds = (predictions > 0.5).astype(int).flatten()
        
        # Calculate metrics
        print("\n" + "=" * 60)
        print(" TEST RESULTS ")
        print("=" * 60)
        
        print("\nClassification Report:")
        print(classification_report(valid_labels, binary_preds, 
                                   target_names=['Real', 'AI-Generated']))
        
        # Confusion matrix
        cm = confusion_matrix(valid_labels, binary_preds)
        
        # Calculate accuracy
        accuracy = np.sum(binary_preds == valid_labels) / len(valid_labels)
        print(f"\n✅ Test Accuracy: {accuracy:.2%}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks([0.5, 1.5], ['Real', 'AI-Generated'])
        plt.yticks([0.5, 1.5], ['Real', 'AI-Generated'])
        plt.tight_layout()
        plt.savefig('confusion_matrix_optimized.png', dpi=100)
        print("\n✅ Confusion matrix saved to confusion_matrix_optimized.png")
        
        return accuracy
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Train')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Train')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history_optimized.png', dpi=100)
        print("✅ Training history saved to training_history_optimized.png")
    
    def run_full_training(self, max_images=1000, epochs=30, batch_size=8):
        """Run complete training pipeline"""
        print("=" * 60)
        print(" AI IMAGE DETECTION - OPTIMIZED TRAINING ")
        print(" Dell Latitude E5470 (CPU Only) ")
        print("=" * 60)
        print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Prepare dataset
            self.prepare_dataset(max_images_per_class=max_images)
            
            # Build model
            self.build_model()
            
            # Train
            if not self.train(epochs=epochs, batch_size=batch_size):
                return False
            
            # Evaluate
            accuracy = self.evaluate()
            
            # Plot results
            self.plot_training_history()
            
            print("\n" + "=" * 60)
            print(" ✅ TRAINING COMPLETED SUCCESSFULLY ")
            print("=" * 60)
            print(f"\nFinal Test Accuracy: {accuracy:.2%}")
            print(f"\nModel saved to: model.h5")
            print(f"Best model saved to: best_model_optimized.h5")
            print(f"\n🚀 Ready to use! Run:")
            print("   python app.py  (start backend)")
            print("   cd frontend && npm run dev  (start frontend)")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized AI Image Detection Training')
    parser.add_argument('--max_images', type=int, default=1000,
                       help='Maximum images per class')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (keep low for 16GB RAM)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = OptimizedTrainer()
    
    # Run training
    success = trainer.run_full_training(
        max_images=args.max_images,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    if not success:
        print("\n❌ Training failed. Check errors above.")
        exit(1)


if __name__ == "__main__":
    main()
