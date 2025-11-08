"""Unit tests for models module."""
import pytest
import tensorflow as tf

from fedgan.config import ModelConfig
from fedgan.models import (
    DCGANFactory,
    ModelFactory,
    build_discriminator,
    build_generator,
    get_model_factory,
)


class TestModelArchitectures:
    """Tests for model architecture builders."""
    
    def test_build_generator(self):
        """Test building generator model."""
        generator = build_generator(latent_dim=100)
        
        assert generator.name == 'generator'
        assert len(generator.inputs) == 1
        assert generator.input_shape == (None, 100)
    
    def test_generator_output_shape(self):
        """Test generator produces correct output shape."""
        generator = build_generator(latent_dim=200)
        
        # Test with batch of noise vectors
        noise = tf.random.normal([4, 200])
        output = generator(noise, training=False)
        
        assert output.shape == (4, 128, 128, 1)
    
    def test_build_discriminator(self):
        """Test building discriminator model."""
        discriminator = build_discriminator(image_shape=(128, 128, 1))
        
        assert discriminator.name == 'discriminator'
        assert len(discriminator.inputs) == 1
        assert discriminator.input_shape == (None, 128, 128, 1)
    
    def test_discriminator_output_shape(self):
        """Test discriminator produces correct output shape."""
        discriminator = build_discriminator()
        
        # Test with batch of images
        images = tf.random.normal([4, 128, 128, 1])
        output = discriminator(images, training=False)
        
        assert output.shape == (4, 1)
        # Output should be probabilities
        assert tf.reduce_all(output >= 0.0)
        assert tf.reduce_all(output <= 1.0)
    
    def test_custom_generator_filters(self):
        """Test generator with custom filter configuration."""
        custom_filters = [512, 256, 128]
        generator = build_generator(
            latent_dim=100,
            filters=custom_filters
        )
        
        # Should still produce valid model
        noise = tf.random.normal([2, 100])
        output = generator(noise, training=False)
        assert output.shape[0] == 2
    
    def test_custom_discriminator_filters(self):
        """Test discriminator with custom filter configuration."""
        custom_filters = [32, 64, 128]
        discriminator = build_discriminator(filters=custom_filters)
        
        images = tf.random.normal([2, 128, 128, 1])
        output = discriminator(images, training=False)
        assert output.shape == (2, 1)


class TestModelFactory:
    """Tests for model factory pattern."""
    
    def test_dcgan_factory_creation(self):
        """Test creating DCGAN factory."""
        config = ModelConfig(latent_dim=200)
        factory = DCGANFactory(config, image_size=128, channels=1)
        
        assert isinstance(factory, ModelFactory)
        assert factory.config == config
    
    def test_dcgan_factory_create_generator(self):
        """Test factory creates generator."""
        config = ModelConfig(latent_dim=150)
        factory = DCGANFactory(config)
        
        generator = factory.create_generator()
        
        assert isinstance(generator, tf.keras.Model)
        assert generator.input_shape == (None, 150)
    
    def test_dcgan_factory_create_discriminator(self):
        """Test factory creates discriminator."""
        config = ModelConfig()
        factory = DCGANFactory(config, image_size=128)
        
        discriminator = factory.create_discriminator()
        
        assert isinstance(discriminator, tf.keras.Model)
        assert discriminator.input_shape == (None, 128, 128, 1)
    
    def test_dcgan_factory_create_models(self):
        """Test factory creates both models."""
        config = ModelConfig()
        factory = DCGANFactory(config)
        
        generator, discriminator = factory.create_models()
        
        assert isinstance(generator, tf.keras.Model)
        assert isinstance(discriminator, tf.keras.Model)
    
    def test_get_model_factory_dcgan(self):
        """Test getting factory by name."""
        config = ModelConfig()
        factory = get_model_factory('dcgan', config, image_size=128)
        
        assert isinstance(factory, DCGANFactory)
    
    def test_get_model_factory_simple(self):
        """Test getting simple factory."""
        factory = get_model_factory('simple', latent_dim=100)
        
        assert isinstance(factory, ModelFactory)
    
    def test_get_model_factory_unknown(self):
        """Test error with unknown factory."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            get_model_factory('nonexistent')
    
    def test_get_model_factory_dcgan_no_config(self):
        """Test error when DCGAN factory created without config."""
        with pytest.raises(ValueError, match="ModelConfig required"):
            get_model_factory('dcgan')


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for models."""
    
    def test_generator_discriminator_compatibility(self):
        """Test that generator output can be fed to discriminator."""
        config = ModelConfig(latent_dim=200)
        factory = DCGANFactory(config, image_size=128)
        
        generator = factory.create_generator()
        discriminator = factory.create_discriminator()
        
        # Generate fake images
        noise = tf.random.normal([8, 200])
        fake_images = generator(noise, training=False)
        
        # Discriminate fake images
        predictions = discriminator(fake_images, training=False)
        
        assert predictions.shape == (8, 1)
        assert tf.reduce_all(predictions >= 0.0)
        assert tf.reduce_all(predictions <= 1.0)
    
    def test_model_trainable_variables(self):
        """Test that models have trainable variables."""
        config = ModelConfig()
        factory = DCGANFactory(config)
        
        generator = factory.create_generator()
        discriminator = factory.create_discriminator()
        
        assert len(generator.trainable_variables) > 0
        assert len(discriminator.trainable_variables) > 0
    
    def test_model_serialization(self):
        """Test that models can be saved and loaded."""
        import tempfile
        from pathlib import Path
        
        config = ModelConfig(latent_dim=100)
        factory = DCGANFactory(config)
        generator = factory.create_generator()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "generator.keras"
            generator.save(save_path)
            
            loaded_generator = tf.keras.models.load_model(save_path)
            
            # Test that loaded model produces same output
            noise = tf.random.normal([2, 100])
            tf.random.set_seed(42)
            original_output = generator(noise, training=False)
            tf.random.set_seed(42)
            loaded_output = loaded_generator(noise, training=False)
            
            # Outputs should be identical
            tf.debugging.assert_near(original_output, loaded_output, atol=1e-5)
