"""
Evaluation metrics for GAN models using Strategy pattern.

This module provides different metric strategies for evaluating
the quality of generated images.
"""
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from scipy import linalg


class MetricStrategy(ABC):
    """Abstract strategy for evaluation metrics.
    
    This implements the Strategy pattern for interchangeable
    evaluation metrics.
    """
    
    @abstractmethod
    def compute(self, real_images: tf.Tensor, fake_images: tf.Tensor) -> float:
        """Compute the metric.
        
        Args:
            real_images: Real image samples.
            fake_images: Generated image samples.
        
        Returns:
            Metric value.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the metric name."""
        pass


class FIDMetric(MetricStrategy):
    """Fréchet Inception Distance (FID) metric.
    
    FID measures the distance between feature distributions of
    real and generated images using Inception features.
    
    Args:
        inception_model: InceptionV3 model for feature extraction.
    """
    
    def __init__(self, inception_model=None):
        if inception_model is None:
            # Load InceptionV3 without top layers
            inception_model = tf.keras.applications.InceptionV3(
                include_top=False,
                pooling='avg',
                input_shape=(299, 299, 3)
            )
        self.inception_model = inception_model
    
    @property
    def name(self) -> str:
        return "FID"
    
    def compute(self, real_images: tf.Tensor, fake_images: tf.Tensor) -> float:
        """Calculate FID score.
        
        Args:
            real_images: Real images tensor.
            fake_images: Generated images tensor.
        
        Returns:
            FID score (lower is better).
        """
        # Extract features
        real_features = self._extract_features(real_images)
        fake_features = self._extract_features(fake_images)
        
        # Calculate statistics
        mu_real, sigma_real = self._calculate_statistics(real_features)
        mu_fake, sigma_fake = self._calculate_statistics(fake_features)
        
        # Calculate FID
        return self._calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
    
    def _extract_features(self, images: tf.Tensor) -> np.ndarray:
        """Extract Inception features from images.
        
        Args:
            images: Images tensor.
        
        Returns:
            Feature array.
        """
        # Resize to Inception input size
        resized = tf.image.resize(images, [299, 299])
        
        # Convert grayscale to RGB if needed
        if images.shape[-1] == 1:
            resized = tf.image.grayscale_to_rgb(resized)
        
        # Normalize to [-1, 1] range (Inception expects this)
        if tf.reduce_max(resized) > 1.0:
            resized = resized / 127.5 - 1.0
        
        # Extract features
        features = self.inception_model.predict(resized, verbose=0)
        return features
    
    @staticmethod
    def _calculate_statistics(features: np.ndarray):
        """Calculate mean and covariance of features.
        
        Args:
            features: Feature array.
        
        Returns:
            Tuple of (mean, covariance).
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    @staticmethod
    def _calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Calculate FID score.
        
        Args:
            mu1, sigma1: Mean and covariance of real features.
            mu2, sigma2: Mean and covariance of fake features.
            eps: Small value for numerical stability.
        
        Returns:
            FID score.
        """
        # Calculate squared difference of means
        diff = mu1 - mu2
        mean_diff = np.sum(diff ** 2)
        
        # Calculate sqrt of product of covariances
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Check for imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # Calculate FID
        fid = mean_diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)


class InceptionScoreMetric(MetricStrategy):
    """Inception Score (IS) metric.
    
    IS measures the quality and diversity of generated images
    using the Inception model's predictions.
    
    Args:
        inception_model: InceptionV3 model with classification head.
        num_splits: Number of splits for computing mean and std.
    """
    
    def __init__(self, inception_model=None, num_splits=10):
        if inception_model is None:
            inception_model = tf.keras.applications.InceptionV3(
                include_top=True,
                input_shape=(299, 299, 3)
            )
        self.inception_model = inception_model
        self.num_splits = num_splits
    
    @property
    def name(self) -> str:
        return "IS"
    
    def compute(self, real_images: tf.Tensor, fake_images: tf.Tensor) -> float:
        """Calculate Inception Score.
        
        Note: IS only uses fake images, real images are ignored.
        
        Args:
            real_images: Not used.
            fake_images: Generated images.
        
        Returns:
            Inception Score (higher is better).
        """
        # Get predictions
        preds = self._get_predictions(fake_images)
        
        # Split into groups
        split_scores = []
        split_size = preds.shape[0] // self.num_splits
        
        for i in range(self.num_splits):
            part = preds[i * split_size: (i + 1) * split_size]
            
            # KL divergence for this split
            py = np.mean(part, axis=0)
            scores = []
            for j in range(part.shape[0]):
                pyx = part[j, :]
                scores.append(self._kl_divergence(pyx, py))
            
            split_scores.append(np.exp(np.mean(scores)))
        
        # Return mean and std
        is_mean = np.mean(split_scores)
        return float(is_mean)
    
    def _get_predictions(self, images: tf.Tensor) -> np.ndarray:
        """Get Inception predictions for images.
        
        Args:
            images: Images tensor.
        
        Returns:
            Prediction probabilities.
        """
        # Resize and convert to RGB
        resized = tf.image.resize(images, [299, 299])
        if images.shape[-1] == 1:
            resized = tf.image.grayscale_to_rgb(resized)
        
        # Normalize
        if tf.reduce_max(resized) > 1.0:
            resized = resized / 127.5 - 1.0
        
        # Get predictions
        preds = self.inception_model.predict(resized, verbose=0)
        return preds
    
    @staticmethod
    def _kl_divergence(p, q, eps=1e-10):
        """Calculate KL divergence between two distributions.
        
        Args:
            p, q: Probability distributions.
            eps: Small value for numerical stability.
        
        Returns:
            KL divergence.
        """
        p = np.asarray(p, dtype=np.float64) + eps
        q = np.asarray(q, dtype=np.float64) + eps
        return np.sum(p * np.log(p / q))


class SSIMMetric(MetricStrategy):
    """Structural Similarity Index (SSIM) metric.
    
    SSIM measures the perceptual similarity between images.
    """
    
    @property
    def name(self) -> str:
        return "SSIM"
    
    def compute(self, real_images: tf.Tensor, fake_images: tf.Tensor) -> float:
        """Calculate mean SSIM.
        
        Args:
            real_images: Real images.
            fake_images: Generated images (same number as real).
        
        Returns:
            Mean SSIM score.
        """
        ssim_values = tf.image.ssim(
            real_images,
            fake_images,
            max_val=2.0  # Images in range [-1, 1]
        )
        return float(tf.reduce_mean(ssim_values))
