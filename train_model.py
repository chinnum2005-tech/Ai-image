"""
Training Pipeline for AI-Generated Image Detection Model
Handles data loading, training, and evaluation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow import keras
import json
from tqdm import tqdm
import argparse
from datetime import datetime

from preprocess import ImagePreprocessor, DataGenerator
from feature_extraction import PixelWiseFeatureExtractor
from model import AIImageDetector, LightweightDetector


class TrainingPipeline:
    """
    Complete training pipeline for AI image detection
    """
    
    def __init__(self, real_dir, fake_dir, model_type='hybrid', use_augmentation=True):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.model_type = model_type
        self.use_augmentation = use_augmentation
        
        # Initialize components
        self.preprocessor = ImagePreprocessor()
        self.feature_extractor = PixelWiseFeatureExtractor()
        self.detector = None
        
        # Data storage
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        # Results storage
        self.history = None
        self.test_results = None
        
    def prepare_data(self):
        """
        Prepare training, validation, and test datasets
        """
        print("=" * 50)
        print("PREPARING DATASET")
        print("=" * 50)
        
        # Load and split dataset
        datasets = self.preprocessor.load_and_split_dataset(
            self.real_dir, 
            self.fake_dir,
            test_split=0.2,
            val_split=0.1
        )
        
        print(f"\nDataset split:")
        print(f"Training: {len(datasets['train'][0])} images")
        print(f"Validation: {len(datasets['val'][0])} images")
        print(f"Test: {len(datasets['test'][0])} images")
        
        # Apply augmentation to training set if enabled
        if self.use_augmentation:
            print("\nApplying data augmentation to training set...")
            train_images, train_labels = self.preprocessor.create_augmented_dataset(
                datasets['train'][0],
                datasets['train'][1],
                augmentation_factor=2
            )
            print(f"Augmented training set: {len(train_images)} images")
        else:
            train_images, train_labels = datasets['train']
        
        # Store datasets
        self.train_data = (train_images, train_labels)
        self.val_data = datasets['val']
        self.test_data = datasets['test']
        
        return True
    
    def extract_pixel_features(self, image_paths, desc="Extracting features"):
        """
        Extract pixel-wise features for a set of images
        """
        features = []
        
        for path in tqdm(image_paths, desc=desc):
            try:
                feature_vector = self.feature_extractor.get_feature_vector(str(path))
                features.append(feature_vector)
            except Exception as e:
                print(f"Error extracting features from {path}: {e}")
                # Use zero features as fallback
                features.append(np.zeros(100))  # Adjust size as needed
        
        return np.array(features)
    
    def prepare_hybrid_data(self):
        """
        Prepare data for hybrid model (images + pixel features)
        """
        print("\n" + "=" * 50)
        print("EXTRACTING PIXEL-WISE FEATURES")
        print("=" * 50)
        
        # Extract features for training set
        print("\nExtracting training features...")
        train_features = self.extract_pixel_features(
            self.train_data[0], 
            "Training features"
        )
        
        # Extract features for validation set
        print("\nExtracting validation features...")
        val_features = self.extract_pixel_features(
            self.val_data[0],
            "Validation features"
        )
        
        # Extract features for test set
        print("\nExtracting test features...")
        test_features = self.extract_pixel_features(
            self.test_data[0],
            "Test features"
        )
        
        # Normalize features
        print("\nNormalizing features...")
        train_features, mean, std = self.preprocessor.normalize_features(train_features)
        val_features = self.preprocessor.apply_normalization(val_features, mean, std)
        test_features = self.preprocessor.apply_normalization(test_features, mean, std)
        
        # Save normalization parameters
        np.save('feature_mean.npy', mean)
        np.save('feature_std.npy', std)
        
        return train_features, val_features, test_features
    
    def create_tf_dataset(self, images, labels, features=None, batch_size=32, augment=False):
        """
        Create TensorFlow dataset for training
        """
        def load_and_preprocess(path, label):
            # Read and preprocess image
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3)
            image = tf.image.resize(image, [256, 256])
            image = tf.cast(image, tf.float32) / 255.0
            return image, label
        
        # Convert paths to strings
        image_paths = [str(p) for p in images]
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Add pixel features if provided
        if features is not None:
            feature_dataset = tf.data.Dataset.from_tensor_slices(features)
            dataset = tf.data.Dataset.zip((dataset, feature_dataset))
            dataset = dataset.map(
                lambda x, f: ((x[0], f), x[1]),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Shuffle and batch
        if augment:
            dataset = dataset.shuffle(1000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def build_model(self):
        """
        Build and compile the model
        """
        print("\n" + "=" * 50)
        print("BUILDING MODEL")
        print("=" * 50)
        
        if self.model_type == 'hybrid':
            print("Building hybrid model (CNN + Pixel Features)...")
            self.detector = AIImageDetector(num_pixel_features=len(self.feature_extractor.feature_names) or 100)
            self.detector.build_hybrid_model()
        elif self.model_type == 'cnn':
            print("Building CNN-only model...")
            self.detector = AIImageDetector()
            self.detector.build_cnn_only_model()
        else:  # lightweight
            print("Building lightweight model...")
            self.detector = LightweightDetector()
            self.detector.build_model()
        
        # Compile model
        self.detector.compile_model(learning_rate=0.001)
        
        print(f"\nModel architecture: {self.model_type}")
        print(f"Total parameters: {self.detector.model.count_params():,}")
        
        return True
    
    def train(self, epochs=50, batch_size=32):
        """
        Train the model
        """
        print("\n" + "=" * 50)
        print("TRAINING MODEL")
        print("=" * 50)
        
        # Prepare TensorFlow datasets
        if self.model_type == 'hybrid':
            # Extract pixel features
            train_features, val_features, test_features = self.prepare_hybrid_data()
            
            # Create datasets with features
            train_dataset = self.create_tf_dataset(
                self.train_data[0], self.train_data[1], 
                train_features, batch_size, augment=True
            )
            val_dataset = self.create_tf_dataset(
                self.val_data[0], self.val_data[1],
                val_features, batch_size, augment=False
            )
            self.test_dataset = self.create_tf_dataset(
                self.test_data[0], self.test_data[1],
                test_features, batch_size, augment=False
            )
        else:
            # Create datasets without features
            train_dataset = self.create_tf_dataset(
                self.train_data[0], self.train_data[1],
                None, batch_size, augment=True
            )
            val_dataset = self.create_tf_dataset(
                self.val_data[0], self.val_data[1],
                None, batch_size, augment=False
            )
            self.test_dataset = self.create_tf_dataset(
                self.test_data[0], self.test_data[1],
                None, batch_size, augment=False
            )
        
        # Get callbacks
        callbacks = self.detector.get_callbacks(
            checkpoint_path=f'best_model_{self.model_type}.h5',
            patience=10
        )
        
        # Train model
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Batch size: {batch_size}")
        
        self.history = self.detector.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        
        return self.history
    
    def evaluate(self):
        """
        Evaluate model on test set
        """
        print("\n" + "=" * 50)
        print("EVALUATING MODEL")
        print("=" * 50)
        
        # Evaluate on test set
        test_results = self.detector.model.evaluate(self.test_dataset, verbose=1)
        
        # Store results
        self.test_results = {}
        for i, metric_name in enumerate(self.detector.model.metrics_names):
            self.test_results[metric_name] = test_results[i]
        
        print("\nTest Results:")
        for metric, value in self.test_results.items():
            print(f"{metric}: {value:.4f}")
        
        # Generate predictions for detailed analysis
        predictions = []
        true_labels = []
        
        for batch in self.test_dataset:
            if self.model_type == 'hybrid':
                batch_preds = self.detector.model.predict(batch[0], verbose=0)
            else:
                batch_preds = self.detector.model.predict(batch[0], verbose=0)
            
            predictions.extend(batch_preds.flatten())
            true_labels.extend(batch[1].numpy())
        
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Binary predictions
        binary_preds = (predictions > 0.5).astype(int)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, binary_preds, 
                                   target_names=['Real', 'AI-Generated']))
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, binary_preds)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(true_labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        # Store additional metrics
        self.test_results['predictions'] = predictions
        self.test_results['true_labels'] = true_labels
        self.test_results['confusion_matrix'] = cm
        self.test_results['roc_curve'] = (fpr, tpr, roc_auc)
        
        return self.test_results
    
    def plot_results(self):
        """
        Plot training history and evaluation results
        """
        print("\n" + "=" * 50)
        print("GENERATING PLOTS")
        print("=" * 50)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot training history
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC
        if 'auc' in self.history.history:
            axes[0, 2].plot(self.history.history['auc'], label='Train')
            axes[0, 2].plot(self.history.history['val_auc'], label='Validation')
            axes[0, 2].set_title('Model AUC')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('AUC')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        
        # Confusion Matrix
        cm = self.test_results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_xticklabels(['Real', 'AI-Generated'])
        axes[1, 0].set_yticklabels(['Real', 'AI-Generated'])
        
        # ROC Curve
        fpr, tpr, roc_auc = self.test_results['roc_curve']
        axes[1, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1, 1].set_xlim([0.0, 1.0])
        axes[1, 1].set_ylim([0.0, 1.05])
        axes[1, 1].set_xlabel('False Positive Rate')
        axes[1, 1].set_ylabel('True Positive Rate')
        axes[1, 1].set_title('ROC Curve')
        axes[1, 1].legend(loc="lower right")
        axes[1, 1].grid(True)
        
        # Prediction Distribution
        axes[1, 2].hist(self.test_results['predictions'][self.test_results['true_labels']==0], 
                       bins=30, alpha=0.5, label='Real', color='blue')
        axes[1, 2].hist(self.test_results['predictions'][self.test_results['true_labels']==1], 
                       bins=30, alpha=0.5, label='AI-Generated', color='red')
        axes[1, 2].set_xlabel('Prediction Score')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Prediction Distribution')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        print("Plots saved to training_results.png")
    
    def save_model(self):
        """
        Save the trained model and metadata
        """
        print("\n" + "=" * 50)
        print("SAVING MODEL")
        print("=" * 50)
        
        # Save model
        model_path = 'model.h5'
        self.detector.save_model(model_path)
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'input_shape': [256, 256, 3],
            'num_pixel_features': len(self.feature_extractor.feature_names) if self.model_type == 'hybrid' else 0,
            'training_date': datetime.now().isoformat(),
            'test_results': {
                'accuracy': float(self.test_results['accuracy']),
                'auc': float(self.test_results['auc']) if 'auc' in self.test_results else None,
                'loss': float(self.test_results['loss'])
            }
        }
        
        with open('model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Model saved to {model_path}")
        print("Metadata saved to model_metadata.json")
    
    def run_pipeline(self, epochs=50, batch_size=32):
        """
        Run the complete training pipeline
        """
        print("\n" + "=" * 70)
        print(" AI-GENERATED IMAGE DETECTION - TRAINING PIPELINE ")
        print("=" * 70)
        
        # Step 1: Prepare data
        self.prepare_data()
        
        # Step 2: Build model
        self.build_model()
        
        # Step 3: Train model
        self.train(epochs, batch_size)
        
        # Step 4: Evaluate model
        self.evaluate()
        
        # Step 5: Plot results
        self.plot_results()
        
        # Step 6: Save model
        self.save_model()
        
        print("\n" + "=" * 70)
        print(" TRAINING PIPELINE COMPLETED SUCCESSFULLY! ")
        print("=" * 70)
        
        return True


def main():
    """
    Main function to run training
    """
    parser = argparse.ArgumentParser(description='Train AI Image Detection Model')
    parser.add_argument('--real_dir', type=str, default='dataset/real',
                       help='Directory containing real images')
    parser.add_argument('--fake_dir', type=str, default='dataset/fake',
                       help='Directory containing AI-generated images')
    parser.add_argument('--model_type', type=str, default='hybrid',
                       choices=['hybrid', 'cnn', 'lightweight'],
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--no_augmentation', action='store_true',
                       help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = TrainingPipeline(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        model_type=args.model_type,
        use_augmentation=not args.no_augmentation
    )
    
    # Run training
    pipeline.run_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    # For testing without command line args
    pipeline = TrainingPipeline(
        real_dir='dataset/real',
        fake_dir='dataset/fake',
        model_type='hybrid',
        use_augmentation=True
    )
    
    print("Training pipeline initialized successfully!")
    print("Run with appropriate dataset directories to start training.")
