"""
Deep Learning Model Architecture for AI-Generated Image Detection
Implements hybrid CNN + Feature Fusion approach
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import numpy as np


class AIImageDetector:
    """
    Hybrid model combining CNN and pixel-wise features
    """
    
    def __init__(self, input_shape=(256, 256, 3), num_pixel_features=100):
        self.input_shape = input_shape
        self.num_pixel_features = num_pixel_features
        self.model = None
        self.feature_model = None
        self.history = None
        
    def build_cnn_branch(self):
        """
        Build CNN branch for image processing
        """
        # Input layer
        input_img = layers.Input(shape=self.input_shape, name='image_input')
        
        # Initial convolution block
        x = layers.Conv2D(32, (3, 3), padding='same')(input_img)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Second convolution block
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Third convolution block
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Fourth convolution block
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Attention mechanism
        attention = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
        x = layers.Multiply()([x, attention])
        
        # Global feature extraction
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(256)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        
        cnn_features = layers.Dense(128, name='cnn_features')(x)
        
        return input_img, cnn_features
    
    def build_hybrid_model(self):
        """
        Build hybrid model combining CNN and pixel-wise features
        """
        # CNN branch
        input_img, cnn_features = self.build_cnn_branch()
        
        # Pixel-wise features branch
        input_pixel = layers.Input(shape=(self.num_pixel_features,), name='pixel_features')
        
        # Process pixel features
        x_pixel = layers.Dense(64)(input_pixel)
        x_pixel = layers.BatchNormalization()(x_pixel)
        x_pixel = layers.Activation('relu')(x_pixel)
        x_pixel = layers.Dropout(0.3)(x_pixel)
        
        x_pixel = layers.Dense(32)(x_pixel)
        x_pixel = layers.BatchNormalization()(x_pixel)
        x_pixel = layers.Activation('relu')(x_pixel)
        
        # Fusion layer
        combined = layers.Concatenate()([cnn_features, x_pixel])
        
        # Final classification layers
        x = layers.Dense(128)(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(64)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        self.model = models.Model(
            inputs=[input_img, input_pixel],
            outputs=output,
            name='AI_Image_Detector'
        )
        
        return self.model
    
    def build_cnn_only_model(self):
        """
        Build CNN-only model (without pixel features)
        """
        # CNN branch
        input_img, cnn_features = self.build_cnn_branch()
        
        # Classification layers
        x = layers.Dense(64)(cnn_features)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        self.model = models.Model(
            inputs=input_img,
            outputs=output,
            name='AI_Image_Detector_CNN'
        )
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_hybrid_model() or build_cnn_only_model() first.")
        
        # Use Adam optimizer with learning rate scheduling
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        return self.model
    
    def get_callbacks(self, checkpoint_path='best_model.h5', patience=10):
        """
        Get training callbacks
        """
        callbacks_list = [
            # Save best model
            callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_auc',
                mode='max',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks_list
    
    def train(self, train_data, val_data, epochs=50, batch_size=32):
        """
        Train the model
        """
        if self.model is None:
            raise ValueError("Model not compiled yet.")
        
        # Get callbacks
        callbacks_list = self.get_callbacks()
        
        # Train model
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return self.history
    
    def predict(self, image, pixel_features=None):
        """
        Predict whether image is AI-generated
        """
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        # Prepare input
        if pixel_features is not None:
            # Hybrid model
            prediction = self.model.predict([image, pixel_features])
        else:
            # CNN-only model
            prediction = self.model.predict(image)
        
        return prediction
    
    def evaluate(self, test_data):
        """
        Evaluate model on test data
        """
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        results = self.model.evaluate(test_data, verbose=1)
        
        # Create results dictionary
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = results[i]
        
        return metrics
    
    def save_model(self, filepath='model.h5'):
        """
        Save the trained model
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='model.h5'):
        """
        Load a trained model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model
    
    def get_model_summary(self):
        """
        Get model summary
        """
        if self.model is None:
            raise ValueError("Model not built yet.")
        
        return self.model.summary()


class LightweightDetector:
    """
    Lightweight model for faster inference
    """
    
    def __init__(self, input_shape=(256, 256, 3)):
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self):
        """
        Build lightweight EfficientNet-based model
        """
        # Use MobileNetV2 as base (lightweight and efficient)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base layers initially
        base_model.trainable = False
        
        # Build model
        inputs = keras.Input(shape=self.input_shape)
        
        # Preprocess for MobileNetV2
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        
        # Base model
        x = base_model(x, training=False)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Classification head
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return self.model
    
    def unfreeze_base(self, num_layers=20):
        """
        Unfreeze top layers of base model for fine-tuning
        """
        base_model = self.model.layers[3]  # Get base model layer
        base_model.trainable = True
        
        # Freeze all layers except top num_layers
        for layer in base_model.layers[:-num_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.compile_model(learning_rate=0.0001)


if __name__ == "__main__":
    # Test model creation
    detector = AIImageDetector()
    model = detector.build_hybrid_model()
    detector.compile_model()
    
    print("Hybrid AI Image Detector Model created successfully!")
    print(f"Model has {model.count_params():,} parameters")
    
    # Test lightweight model
    lightweight = LightweightDetector()
    lightweight.build_model()
    lightweight.compile_model()
    
    print("\nLightweight Detector Model created successfully!")
    print(f"Model has {lightweight.model.count_params():,} parameters")
