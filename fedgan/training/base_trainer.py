"""
Base trainer implementing Template Method pattern.

This module provides an abstract base trainer that defines the
structure of the training loop while allowing subclasses to customize
specific training steps.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import tensorflow as tf

from fedgan.config import ExperimentConfig
from fedgan.training.callbacks import CallbackList, TrainingCallback


class BaseTrainer(ABC):
    """Abstract base trainer using Template Method pattern.
    
    This class defines the skeleton of the training algorithm,
    with specific steps implemented by subclasses.
    
    Args:
        config: Experiment configuration.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.callbacks = CallbackList()
        self.epoch = 0
        self.global_step = 0
    
    def add_callback(self, callback: TrainingCallback) -> None:
        """Add a training callback.
        
        Args:
            callback: Callback instance to add.
        """
        self.callbacks.append(callback)
    
    def train(self, dataset: tf.data.Dataset, epochs: int) -> Dict[str, Any]:
        """Train the model (template method).
        
        This method defines the overall training structure.
        Subclasses implement specific steps.
        
        Args:
            dataset: Training dataset.
            epochs: Number of epochs to train.
        
        Returns:
            Dictionary of training history/metrics.
        """
        history = {'losses': [], 'metrics': {}}
        
        self.callbacks.on_train_begin()
        
        for epoch in range(epochs):
            self.epoch = epoch
            self.callbacks.on_epoch_begin(epoch)
            
            epoch_logs = self._train_epoch(dataset)
            history['losses'].append(epoch_logs)
            
            self.callbacks.on_epoch_end(epoch, epoch_logs)
        
        self.callbacks.on_train_end()
        
        return history
    
    def _train_epoch(self, dataset: tf.data.Dataset) -> Dict[str, Any]:
        """Train for one epoch.
        
        Args:
            dataset: Training dataset.
        
        Returns:
            Dictionary of epoch metrics.
        """
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataset):
            self.callbacks.on_batch_begin(batch_idx)
            
            batch_metrics = self.train_step(batch)
            epoch_losses.append(batch_metrics)
            
            self.callbacks.on_batch_end(batch_idx, batch_metrics)
            self.global_step += 1
            
            # Break after one epoch worth of data
            if hasattr(self, 'steps_per_epoch'):
                if batch_idx >= self.steps_per_epoch:
                    break
        
        # Aggregate metrics
        return self._aggregate_metrics(epoch_losses)
    
    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, Any]:
        """Execute one training step.
        
        This method must be implemented by subclasses.
        
        Args:
            batch: Batch of training data.
        
        Returns:
            Dictionary of metrics for this step.
        """
        pass
    
    def _aggregate_metrics(self, metrics_list: list) -> Dict[str, Any]:
        """Aggregate metrics across batches.
        
        Args:
            metrics_list: List of metric dictionaries from each batch.
        
        Returns:
            Aggregated metrics.
        """
        if not metrics_list:
            return {}
        
        # Get all metric names
        keys = metrics_list[0].keys()
        aggregated = {}
        
        for key in keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                aggregated[key] = float(tf.reduce_mean(values))
        
        return aggregated
    
    @abstractmethod
    def get_models(self) -> Dict[str, tf.keras.Model]:
        """Get trained models.
        
        Returns:
            Dictionary mapping model names to model instances.
        """
        pass


class GANTrainer(BaseTrainer):
    """Trainer for GAN models.
    
    Args:
        config: Experiment configuration.
        generator: Generator model.
        discriminator: Discriminator model.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        generator: tf.keras.Model,
        discriminator: tf.keras.Model
    ):
        super().__init__(config)
        self.generator = generator
        self.discriminator = discriminator
        
        # Create optimizers
        self.gen_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.model.learning_rate,
            beta_1=config.model.beta_1,
            beta_2=config.model.beta_2
        )
        self.disc_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.model.learning_rate,
            beta_1=config.model.beta_1,
            beta_2=config.model.beta_2
        )
        
        # Loss functions
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
    @tf.function
    def train_step(self, batch: tf.Tensor) -> Dict[str, Any]:
        """Execute one GAN training step.
        
        Args:
            batch: Batch of real images.
        
        Returns:
            Dictionary with generator and discriminator losses.
        """
        real_images = batch
        batch_size = tf.shape(real_images)[0]
        
        # Generate noise
        noise = tf.random.normal([batch_size, self.config.model.latent_dim])
        
        # Train discriminator
        with tf.GradientTape() as disc_tape:
            # Generate fake images
            fake_images = self.generator(noise, training=True)
            
            # Get discriminator predictions
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            
            # Calculate discriminator loss
            # Real images should be classified as 1 (with label smoothing)
            real_loss = self.bce(tf.ones_like(real_output) * 0.9, real_output)
            # Fake images should be classified as 0
            fake_loss = self.bce(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss
        
        # Update discriminator
        disc_gradients = disc_tape.gradient(
            disc_loss,
            self.discriminator.trainable_variables
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )
        
        # Train generator
        with tf.GradientTape() as gen_tape:
            # Generate new fake images
            noise = tf.random.normal([batch_size, self.config.model.latent_dim])
            fake_images = self.generator(noise, training=True)
            
            # Get discriminator prediction on fake images
            fake_output = self.discriminator(fake_images, training=True)
            
            # Generator wants discriminator to classify fakes as real
            gen_loss = self.bce(tf.ones_like(fake_output), fake_output)
        
        # Update generator
        gen_gradients = gen_tape.gradient(
            gen_loss,
            self.generator.trainable_variables
        )
        self.gen_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )
        
        return {
            'gen_loss': gen_loss,
            'disc_loss': disc_loss,
            'real_acc': tf.reduce_mean(tf.cast(real_output > 0.5, tf.float32)),
            'fake_acc': tf.reduce_mean(tf.cast(fake_output < 0.5, tf.float32))
        }
    
    def get_models(self) -> Dict[str, tf.keras.Model]:
        """Get generator and discriminator models.
        
        Returns:
            Dictionary with 'generator' and 'discriminator'.
        """
        return {
            'generator': self.generator,
            'discriminator': self.discriminator
        }
