"""Unit tests for configuration module."""
import tempfile
from pathlib import Path

import pytest
import yaml

from fedgan.config import (
    ConfigBuilder,
    ConfigRegistry,
    DataConfig,
    ExperimentConfig,
    FederatedConfig,
    ModelConfig,
    PathConfig,
    PresetConfigBuilder,
    get_config,
    set_config,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.latent_dim == 200
        assert config.learning_rate == 0.0002
        assert config.beta_1 == 0.5
        assert config.use_batch_norm is True
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModelConfig(latent_dim=100, learning_rate=0.001)
        assert config.latent_dim == 100
        assert config.learning_rate == 0.001
    
    def test_validation_success(self):
        """Test successful validation."""
        config = ModelConfig()
        config.validate()  # Should not raise
    
    def test_validation_invalid_latent_dim(self):
        """Test validation fails with invalid latent_dim."""
        config = ModelConfig(latent_dim=-10)
        with pytest.raises(ValueError, match="latent_dim must be positive"):
            config.validate()
    
    def test_validation_invalid_learning_rate(self):
        """Test validation fails with invalid learning_rate."""
        config = ModelConfig(learning_rate=1.5)
        with pytest.raises(ValueError, match="learning_rate must be between"):
            config.validate()


class TestDataConfig:
    """Tests for DataConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DataConfig()
        assert config.image_size == 128
        assert config.channels == 1
        assert config.batch_size == 16
        assert config.clip_limit == 2.0
    
    def test_validation_success(self):
        """Test successful validation."""
        config = DataConfig()
        config.validate()  # Should not raise
    
    def test_validation_image_size_not_divisible_by_16(self):
        """Test validation fails if image_size not divisible by 16."""
        config = DataConfig(image_size=100)
        with pytest.raises(ValueError, match="image_size must be divisible by 16"):
            config.validate()
    
    def test_validation_invalid_channels(self):
        """Test validation fails with invalid channels."""
        config = DataConfig(channels=2)
        with pytest.raises(ValueError, match="channels must be 1"):
            config.validate()


class TestFederatedConfig:
    """Tests for FederatedConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = FederatedConfig()
        assert config.num_clients == 5
        assert config.local_epochs == 5
        assert config.federated_rounds == 2
        assert config.aggregation_strategy == "fedavg"
    
    def test_validation_success(self):
        """Test successful validation."""
        config = FederatedConfig()
        config.validate()  # Should not raise
    
    def test_validation_invalid_aggregation_strategy(self):
        """Test validation fails with unknown aggregation strategy."""
        config = FederatedConfig(aggregation_strategy="unknown")
        with pytest.raises(ValueError, match="Unknown aggregation strategy"):
            config.validate()


class TestPathConfig:
    """Tests for PathConfig dataclass."""
    
    def test_default_values(self):
        """Test default path values."""
        config = PathConfig()
        assert config.data_dir == Path("data")
        assert config.model_dir == Path("models")
    
    def test_string_to_path_conversion(self):
        """Test that strings are converted to Path objects."""
        config = PathConfig(data_dir="custom/data", model_dir="custom/models")
        assert isinstance(config.data_dir, Path)
        assert isinstance(config.model_dir, Path)
    
    def test_create_directories(self, tmp_path):
        """Test directory creation."""
        config = PathConfig(
            data_dir=tmp_path / "data",
            model_dir=tmp_path / "models",
            log_dir=tmp_path / "logs",
            output_dir=tmp_path / "outputs",
            checkpoint_dir=tmp_path / "checkpoints",
        )
        config.create_directories()
        
        assert config.data_dir.exists()
        assert config.model_dir.exists()
        assert config.log_dir.exists()


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""
    
    def test_default_creation(self):
        """Test creating config with minimal parameters."""
        config = ExperimentConfig(name="test_experiment")
        assert config.name == "test_experiment"
        assert config.seed == 42
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.data, DataConfig)
    
    def test_validation_success(self):
        """Test successful validation."""
        config = ExperimentConfig(name="test")
        config.validate()  # Should not raise
    
    def test_validation_empty_name(self):
        """Test validation fails with empty name."""
        config = ExperimentConfig(name="")
        with pytest.raises(ValueError, match="name cannot be empty"):
            config.validate()
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "name": "test_experiment",
            "description": "Test description",
            "seed": 123,
            "model": {"latent_dim": 150},
            "data": {"batch_size": 32},
            "federated": {"num_clients": 3},
        }
        
        config = ExperimentConfig.from_dict(config_dict)
        
        assert config.name == "test_experiment"
        assert config.seed == 123
        assert config.model.latent_dim == 150
        assert config.data.batch_size == 32
        assert config.federated.num_clients == 3
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = ExperimentConfig(name="test")
        config_dict = config.to_dict()
        
        assert config_dict["name"] == "test"
        assert "model" in config_dict
        assert "data" in config_dict
        assert "federated" in config_dict
    
    def test_from_yaml(self, tmp_path):
        """Test loading config from YAML file."""
        yaml_content = """
        name: yaml_test
        description: Test from YAML
        seed: 999
        model:
          latent_dim: 150
        data:
          batch_size: 32
        federated:
          num_clients: 7
        """
        
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)
        
        config = ExperimentConfig.from_yaml(str(yaml_file))
        
        assert config.name == "yaml_test"
        assert config.seed == 999
        assert config.model.latent_dim == 150
        assert config.federated.num_clients == 7
    
    def test_from_yaml_file_not_found(self):
        """Test error when YAML file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            ExperimentConfig.from_yaml("nonexistent.yaml")
    
    def test_to_yaml(self, tmp_path):
        """Test saving config to YAML file."""
        config = ExperimentConfig(
            name="test",
            model=ModelConfig(latent_dim=100),
        )
        
        yaml_file = tmp_path / "output.yaml"
        config.to_yaml(str(yaml_file))
        
        assert yaml_file.exists()
        
        # Load and verify
        with open(yaml_file, "r") as f:
            loaded = yaml.safe_load(f)
        
        assert loaded["name"] == "test"
        assert loaded["model"]["latent_dim"] == 100


class TestConfigBuilder:
    """Tests for ConfigBuilder."""
    
    def test_basic_building(self):
        """Test basic configuration building."""
        config = (
            ConfigBuilder()
            .for_experiment("test_exp")
            .with_description("Test description")
            .build()
        )
        
        assert config.name == "test_exp"
        assert config.description == "Test description"
    
    def test_fluent_interface(self):
        """Test fluent interface chaining."""
        config = (
            ConfigBuilder()
            .for_experiment("test")
            .with_seed(123)
            .with_latent_dim(150)
            .with_batch_size(32)
            .with_num_clients(7)
            .with_learning_rate(0.001)
            .build()
        )
        
        assert config.seed == 123
        assert config.model.latent_dim == 150
        assert config.data.batch_size == 32
        assert config.federated.num_clients == 7
        assert config.model.learning_rate == 0.001
    
    def test_with_model_config(self):
        """Test setting model config."""
        config = (
            ConfigBuilder()
            .for_experiment("test")
            .with_model_config(latent_dim=100, learning_rate=0.001)
            .build()
        )
        
        assert config.model.latent_dim == 100
        assert config.model.learning_rate == 0.001
    
    def test_with_metadata(self):
        """Test adding metadata."""
        config = (
            ConfigBuilder()
            .for_experiment("test")
            .with_metadata(author="John Doe", version="1.0")
            .build()
        )
        
        assert config.metadata["author"] == "John Doe"
        assert config.metadata["version"] == "1.0"
    
    def test_validation_on_build(self):
        """Test that build validates the configuration."""
        builder = ConfigBuilder().for_experiment("test")
        builder._model_config = ModelConfig(latent_dim=-10)
        
        with pytest.raises(ValueError):
            builder.build()
    
    def test_reset(self):
        """Test resetting builder to initial state."""
        builder = ConfigBuilder().for_experiment("test1")
        builder.reset()
        
        config = builder.for_experiment("test2").build()
        assert config.name == "test2"


class TestPresetConfigBuilder:
    """Tests for PresetConfigBuilder."""
    
    def test_quick_test_preset(self):
        """Test quick test preset."""
        config = PresetConfigBuilder.for_quick_test().build()
        
        assert config.name == "quick_test"
        assert config.data.image_size == 64
        assert config.data.batch_size == 4
        assert config.federated.num_clients == 2
    
    def test_federated_gan_preset(self):
        """Test federated GAN preset."""
        config = PresetConfigBuilder.for_federated_gan(
            num_clients=10, rounds=20
        ).build()
        
        assert config.federated.num_clients == 10
        assert config.federated.federated_rounds == 20
    
    def test_medical_imaging_preset(self):
        """Test medical imaging preset."""
        config = PresetConfigBuilder.for_medical_imaging().build()
        
        assert config.data.channels == 1  # Grayscale
        assert config.data.image_size == 128


class TestConfigRegistry:
    """Tests for ConfigRegistry singleton."""
    
    def teardown_method(self):
        """Clean up after each test."""
        ConfigRegistry.reset_instance()
    
    def test_singleton_pattern(self):
        """Test that ConfigRegistry is a singleton."""
        registry1 = ConfigRegistry()
        registry2 = ConfigRegistry()
        
        assert registry1 is registry2
    
    def test_set_and_get_config(self):
        """Test setting and getting configuration."""
        config = ExperimentConfig(name="test")
        registry = ConfigRegistry()
        
        registry.set_config(config)
        retrieved = registry.get_config()
        
        assert retrieved is config
        assert retrieved.name == "test"
    
    def test_get_config_not_initialized(self):
        """Test error when getting config before setting."""
        registry = ConfigRegistry()
        
        with pytest.raises(RuntimeError, match="No configuration has been initialized"):
            registry.get_config()
    
    def test_set_config_validates(self):
        """Test that set_config validates the configuration."""
        config = ExperimentConfig(name="")  # Invalid: empty name
        registry = ConfigRegistry()
        
        with pytest.raises(ValueError):
            registry.set_config(config)
    
    def test_has_config(self):
        """Test has_config method."""
        registry = ConfigRegistry()
        
        assert not registry.has_config()
        
        registry.set_config(ExperimentConfig(name="test"))
        
        assert registry.has_config()
    
    def test_clear_config(self):
        """Test clearing configuration."""
        registry = ConfigRegistry()
        registry.set_config(ExperimentConfig(name="test"))
        
        assert registry.has_config()
        
        registry.clear_config()
        
        assert not registry.has_config()
    
    def test_get_or_default(self):
        """Test get_or_default method."""
        registry = ConfigRegistry()
        default = ExperimentConfig(name="default")
        
        # Should return default when not set
        config = registry.get_or_default(default)
        assert config is default
        
        # Should return actual config when set
        actual = ExperimentConfig(name="actual")
        registry.set_config(actual)
        config = registry.get_or_default(default)
        assert config is actual


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def teardown_method(self):
        """Clean up after each test."""
        ConfigRegistry.reset_instance()
    
    def test_get_config(self):
        """Test get_config convenience function."""
        config = ExperimentConfig(name="test")
        set_config(config)
        
        retrieved = get_config()
        assert retrieved is config
    
    def test_set_config(self):
        """Test set_config convenience function."""
        config = ExperimentConfig(name="test")
        set_config(config)
        
        assert ConfigRegistry().has_config()
