"""
Training callbacks using Observer pattern.

This module provides callback classes that observe training events
and perform actions like logging, checkpointing, and visualization.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import tensorflow as tf


class TrainingCallback(ABC):
    """Abstract base class for training callbacks (Observer pattern).
    
    Callbacks observe training events and can perform custom actions
    at different points in the training process.
    """
    
    def on_train_begin(self, logs: Dict[str, Any] = None) -> None:
        """Called when training begins."""
        pass
    
    def on_train_end(self, logs: Dict[str, Any] = None) -> None:
        """Called when training ends."""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """Called at the beginning of an epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """Called at the end of an epoch."""
        pass
    
    def on_batch_begin(self, batch: int, logs: Dict[str, Any] = None) -> None:
        """Called at the beginning of a training batch."""
        pass
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None) -> None:
        """Called at the end of a training batch."""
        pass


class TensorBoardCallback(TrainingCallback):
    """Callback for logging metrics to TensorBoard.
    
    Args:
        log_dir: Directory for TensorBoard logs.
    """
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = tf.summary.create_file_writer(str(self.log_dir))
        self.step = 0
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None) -> None:
        """Log batch metrics to TensorBoard."""
        if logs:
            with self.writer.as_default():
                for name, value in logs.items():
                    if isinstance(value, (int, float)):
                        tf.summary.scalar(name, value, step=self.step)
            self.step += 1
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """Log epoch metrics."""
        if logs:
            with self.writer.as_default():
                for name, value in logs.items():
                    if isinstance(value, (int, float)) and 'epoch' in name.lower():
                        tf.summary.scalar(f"epoch/{name}", value, step=epoch)
    
    def on_train_end(self, logs: Dict[str, Any] = None) -> None:
        """Flush and close writer."""
        self.writer.flush()


class ModelCheckpointCallback(TrainingCallback):
    """Callback for saving model checkpoints.
    
    Args:
        checkpoint_dir: Directory for saving checkpoints.
        save_frequency: Save every N epochs (default: 5).
        save_best_only: Only save when metric improves.
        monitor: Metric to monitor for save_best_only.
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        save_frequency: int = 5,
        save_best_only: bool = False,
        monitor: str = 'loss'
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = save_frequency
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.best_value = float('inf')
        self.models = {}
    
    def set_models(self, **models: tf.keras.Model) -> None:
        """Set models to checkpoint.
        
        Args:
            **models: Named models to save (e.g., generator=gen, discriminator=disc).
        """
        self.models = models
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """Save checkpoint if criteria met."""
        should_save = False
        
        if self.save_best_only and logs and self.monitor in logs:
            current_value = logs[self.monitor]
            if current_value < self.best_value:
                self.best_value = current_value
                should_save = True
        elif (epoch + 1) % self.save_frequency == 0:
            should_save = True
        
        if should_save:
            for name, model in self.models.items():
                save_path = self.checkpoint_dir / f"{name}_epoch_{epoch+1}.keras"
                model.save(save_path)


class ImageGenerationCallback(TrainingCallback):
    """Callback for generating and saving sample images during training.
    
    Args:
        generator: Generator model.
        output_dir: Directory for saving generated images.
        latent_dim: Dimensionality of latent vector.
        num_images: Number of images to generate.
        generation_frequency: Generate every N epochs.
    """
    
    def __init__(
        self,
        generator: tf.keras.Model,
        output_dir: Path,
        latent_dim: int,
        num_images: int = 16,
        generation_frequency: int = 1
    ):
        self.generator = generator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.latent_dim = latent_dim
        self.num_images = num_images
        self.generation_frequency = generation_frequency
        
        # Fixed seed for consistent comparison
        self.seed = tf.random.normal([num_images, latent_dim])
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """Generate and save images."""
        if (epoch + 1) % self.generation_frequency == 0:
            generated = self.generator(self.seed, training=False)
            self._save_image_grid(generated, epoch)
    
    def _save_image_grid(self, images: tf.Tensor, epoch: int) -> None:
        """Save images as a grid."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Denormalize from [-1, 1] to [0, 1]
        images = (images + 1.0) / 2.0
        images = tf.clip_by_value(images, 0.0, 1.0)
        
        # Create grid
        grid_size = int(np.sqrt(self.num_images))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        
        for i, ax in enumerate(axes.flat):
            if i < len(images):
                ax.imshow(images[i, :, :, 0], cmap='gray')
            ax.axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / f"generated_epoch_{epoch+1}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


class CallbackList:
    """Container for managing multiple callbacks.
    
    Args:
        callbacks: List of callback instances.
    """
    
    def __init__(self, callbacks: list = None):
        self.callbacks = callbacks or []
    
    def append(self, callback: TrainingCallback) -> None:
        """Add a callback to the list."""
        self.callbacks.append(callback)
    
    def on_train_begin(self, logs: Dict[str, Any] = None) -> None:
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Dict[str, Any] = None) -> None:
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Dict[str, Any] = None) -> None:
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None) -> None:
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)
