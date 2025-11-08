"""
Model evaluator using Facade pattern.

This module provides a simple interface for comprehensive
model evaluation using multiple metrics.
"""
from pathlib import Path
from typing import Dict, List, Optional

import tensorflow as tf

from fedgan.evaluation.metrics import FIDMetric, InceptionScoreMetric, MetricStrategy


class ModelEvaluator:
    """Facade for comprehensive model evaluation.
    
    This class provides a simple interface for evaluating GAN models
    using multiple metrics.
    
    Args:
        metrics: List of metric strategies to use.
    
    Example:
        >>> from fedgan.evaluation import ModelEvaluator, FIDMetric, InceptionScoreMetric
        >>> metrics = [FIDMetric(), InceptionScoreMetric()]
        >>> evaluator = ModelEvaluator(metrics)
        >>> scores = evaluator.evaluate(generator, real_dataset)
    """
    
    def __init__(self, metrics: Optional[List[MetricStrategy]] = None):
        if metrics is None:
            # Default metrics
            metrics = [FIDMetric(), InceptionScoreMetric()]
        self.metrics = metrics
        self.results = {}
    
    def evaluate(
        self,
        generator: tf.keras.Model,
        real_dataset: tf.data.Dataset,
        num_samples: int = 5000,
        latent_dim: int = 200
    ) -> Dict[str, float]:
        """Evaluate generator against real data.
        
        Args:
            generator: Generator model to evaluate.
            real_dataset: Dataset of real images.
            num_samples: Number of samples to use for evaluation.
            latent_dim: Dimensionality of latent vector.
        
        Returns:
            Dictionary mapping metric names to values.
        """
        # Generate fake images
        fake_images = self._generate_images(generator, num_samples, latent_dim)
        
        # Collect real images
        real_images = self._collect_real_images(real_dataset, num_samples)
        
        # Compute all metrics
        results = {}
        for metric in self.metrics:
            try:
                score = metric.compute(real_images, fake_images)
                results[metric.name] = score
                print(f"{metric.name}: {score:.4f}")
            except Exception as e:
                print(f"Error computing {metric.name}: {e}")
                results[metric.name] = None
        
        self.results = results
        return results
    
    def _generate_images(
        self,
        generator: tf.keras.Model,
        num_samples: int,
        latent_dim: int
    ) -> tf.Tensor:
        """Generate fake images.
        
        Args:
            generator: Generator model.
            num_samples: Number of images to generate.
            latent_dim: Latent dimension.
        
        Returns:
            Generated images tensor.
        """
        noise = tf.random.normal([num_samples, latent_dim])
        generated = generator(noise, training=False)
        return generated
    
    def _collect_real_images(
        self,
        dataset: tf.data.Dataset,
        num_samples: int
    ) -> tf.Tensor:
        """Collect real images from dataset.
        
        Args:
            dataset: Dataset of real images.
            num_samples: Number of samples to collect.
        
        Returns:
            Real images tensor.
        """
        images = []
        for batch in dataset:
            # Handle both batched and unbatched datasets
            if isinstance(batch, dict):
                batch = batch['image']
            
            images.append(batch)
            
            total_collected = sum(b.shape[0] for b in images)
            if total_collected >= num_samples:
                break
        
        # Concatenate and trim to exact number
        all_images = tf.concat(images, axis=0)
        return all_images[:num_samples]
    
    def save_results(self, save_path: str) -> None:
        """Save evaluation results to file.
        
        Args:
            save_path: Path to save results (JSON or CSV).
        """
        import json
        from pathlib import Path
        
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(self.results, f, indent=2)
        else:
            # Save as CSV
            with open(path, 'w') as f:
                f.write("Metric,Value\n")
                for metric, value in self.results.items():
                    f.write(f"{metric},{value}\n")
