"""Integration tests for training components."""
import pytest
import tensorflow as tf

from fedgan.config import ExperimentConfig, ModelConfig
from fedgan.models import DCGANFactory
from fedgan.training import GANTrainer, ImageGenerationCallback, TensorBoardCallback


@pytest.mark.integration
class TestGANTraining:
    """Integration tests for GAN training."""
    
    def test_gan_trainer_basic(self, tmp_path, mock_tfrecord):
        """Test basic GAN training."""
        config = ExperimentConfig(
            name="test",
            model=ModelConfig(latent_dim=100)
        )
        
        factory = DCGANFactory(config.model, image_size=128)
        generator = factory.create_generator()
        discriminator = factory.create_discriminator()
        
        trainer = GANTrainer(config, generator, discriminator)
        
        # Create dataset
        from fedgan.data import DatasetFactory
        data_factory = DatasetFactory(config.data)
        dataset = data_factory.create_training_dataset([str(mock_tfrecord)])
        
        # Train for one epoch
        trainer.steps_per_epoch = 5  # Limit steps
        history = trainer.train(dataset, epochs=1)
        
        assert 'losses' in history
        assert len(history['losses']) > 0
    
    def test_gan_trainer_with_callbacks(self, tmp_path, mock_tfrecord):
        """Test GAN training with callbacks."""
        config = ExperimentConfig(name="test")
        
        factory = DCGANFactory(config.model, image_size=128)
        generator = factory.create_generator()
        discriminator = factory.create_discriminator()
        
        trainer = GANTrainer(config, generator, discriminator)
        
        # Add callbacks
        tb_callback = TensorBoardCallback(tmp_path / "logs")
        img_callback = ImageGenerationCallback(
            generator,
            tmp_path / "images",
            latent_dim=200
        )
        
        trainer.add_callback(tb_callback)
        trainer.add_callback(img_callback)
        
        # Create dataset and train
        from fedgan.data import DatasetFactory
        data_factory = DatasetFactory(config.data)
        dataset = data_factory.create_training_dataset([str(mock_tfrecord)])
        
        trainer.steps_per_epoch = 3
        history = trainer.train(dataset, epochs=2)
        
        assert (tmp_path / "images").exists()
        assert len(list((tmp_path / "images").glob("*.png"))) > 0


@pytest.mark.integration
@pytest.mark.slow
class TestFederatedTraining:
    """Integration tests for federated training."""
    
    def test_federated_round(self, mock_tfrecord):
        """Test one federated round."""
        config = ExperimentConfig(name="test_federated")
        config.federated.num_clients = 2
        config.federated.local_epochs = 1
        
        from fedgan.federated import FederatedTrainer
        from fedgan.models import DCGANFactory
        from fedgan.data import DatasetFactory
        
        factory = DCGANFactory(config.model, image_size=128)
        fed_trainer = FederatedTrainer(config, factory)
        
        # Create client datasets
        data_factory = DatasetFactory(config.data)
        client_datasets = {
            0: data_factory.create_training_dataset([str(mock_tfrecord)]),
            1: data_factory.create_training_dataset([str(mock_tfrecord)])
        }
        
        # Train one round
        results = fed_trainer.train_federated_round(client_datasets)
        
        assert results['num_clients'] == 2
        assert results['round'] == 1
        assert 'client_metrics' in results
