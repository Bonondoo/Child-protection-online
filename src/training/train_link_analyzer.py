"""Train link safety classifier model."""

import os
import yaml
import logging
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LinkAnalyzer:
    """Link safety classifier using URL feature analysis."""

    def __init__(self, config_path: str):
        """Initialize link analyzer.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.scaler = StandardScaler()

    def build_model(self, input_features: int = 20):
        """Build neural network for link classification.
        
        Args:
            input_features: Number of input features from URL analysis
        """
        model = models.Sequential([
            layers.Input(shape=(input_features,)),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
        )
        
        self.model = model
        logger.info("Link analyzer model built successfully")
        self.model.summary()

    def extract_features(self, url_analyses: list) -> np.ndarray:
        """Extract features from URL analysis results.
        
        Args:
            url_analyses: List of URL analysis dictionaries
            
        Returns:
            Feature matrix
        """
        features = []
        
        for analysis in url_analyses:
            feature_vector = [
                len(analysis['url']),  # URL length
                len(analysis['domain']),  # Domain length
                len(analysis['path']),  # Path length
                len(analysis['query_params']),  # Number of parameters
                1 if analysis['has_suspicious_keywords'] else 0,
                len(analysis['suspicious_keywords']),
                1 if analysis['is_suspiciously_long'] else 0,
                1 if analysis['has_excess_special_chars'] else 0,
                1 if analysis['uses_ip_address'] else 0,
                analysis['risk_score'],
                analysis['url'].count('-'),  # Dash count
                analysis['url'].count('_'),  # Underscore count
                analysis['url'].count('@'),  # @ symbol count
                analysis['url'].count('/'),  # Slash count
                analysis['url'].count('?'),  # Question mark count
            ]
            features.append(feature_vector)
        
        return np.array(features)

    def train(self, X_train, y_train, X_val, y_val, epochs: int = 20, batch_size: int = 32):
        """Train the link analyzer model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Model checkpoint
        checkpoint = keras.callbacks.ModelCheckpoint(
            'models/link_analyzer/best_model.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        )
        
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        logger.info("Link analyzer training completed")
        return history

    def evaluate(self, X_test, y_test):
        """Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        X_test_scaled = self.scaler.transform(X_test)
        results = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        
        logger.info(f"Test Loss: {results[0]:.4f}")
        logger.info(f"Test Accuracy: {results[1]:.4f}")
        logger.info(f"Test Precision: {results[2]:.4f}")
        logger.info(f"Test Recall: {results[3]:.4f}")
        logger.info(f"Test AUC: {results[4]:.4f}")
        
        return results

    def predict(self, url_analyses: list):
        """Predict link safety.
        
        Args:
            url_analyses: List of URL analysis results
            
        Returns:
            Predictions and confidence scores
        """
        features = self.extract_features(url_analyses)
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        return predictions

    def save_model(self, filepath: str):
        """Save trained model.
        
        Args:
            filepath: Path to save model
        """
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")


def main(config_path: str):
    """Main training function."""
    logger.info("Starting link analyzer training...")
    
    analyzer = LinkAnalyzer(config_path)
    analyzer.build_model(input_features=15)
    
    # TODO: Load data using DataLoader and LinkPreprocessor
    # link_data = load_link_data()
    # url_analyses = [link_preprocessor.analyze_url(url) for url in link_data]
    # features = analyzer.extract_features(url_analyses)
    # history = analyzer.train(X_train, y_train, X_val, y_val)
    # analyzer.evaluate(X_test, y_test)
    # analyzer.save_model('models/link_analyzer/link_model.h5')
    
    logger.info("Link analyzer training pipeline ready")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/link_model_config.yaml')
    args = parser.parse_args()
    
    main(args.config)
