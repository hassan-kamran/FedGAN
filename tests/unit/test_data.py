"""Unit tests for data module."""
import numpy as np
import pytest
import tensorflow as tf

from fedgan.config import DataConfig, ExperimentConfig
from fedgan.data import (
    CLAHEStep,
    DatasetFactory,
    FederatedDataLoader,
    FlexibleImageParser,
    GammaCorrectionStep,
    LabeledImageParser,
    NormalizationStep,
    ParserFactory,
    PixelBinningStep,
    PreprocessingPipeline,
    UnlabeledImageParser,
)


class TestTFRecordParsers:
    """Tests for TFRecord parsing strategies."""
    
    def test_unlabeled_parser_creation(self):
        """Test creating unlabeled image parser."""
        parser = UnlabeledImageParser(image_size=128, channels=1)
        assert parser.image_size == 128
        assert parser.channels == 1
    
    def test_unlabeled_parser_signature(self):
        """Test output signature for unlabeled parser."""
        parser = UnlabeledImageParser(image_size=128, channels=1)
        sig = parser.output_signature
        
        assert sig.shape == (128, 128, 1)
        assert sig.dtype == tf.float32
    
    def test_labeled_parser_creation(self):
        """Test creating labeled image parser."""
        parser = LabeledImageParser(image_size=128, channels=1, num_classes=5)
        assert parser.image_size == 128
        assert parser.num_classes == 5
    
    def test_labeled_parser_signature(self):
        """Test output signature for labeled parser."""
        parser = LabeledImageParser(image_size=128, channels=1)
        sig = parser.output_signature
        
        assert 'image' in sig
        assert 'label' in sig
        assert sig['image'].shape == (128, 128, 1)
        assert sig['label'].shape == ()
    
    def test_unlabeled_parser_parse(self, mock_tfrecord):
        """Test parsing unlabeled TFRecord."""
        parser = UnlabeledImageParser(image_size=128, channels=1)
        dataset = tf.data.TFRecordDataset(str(mock_tfrecord))
        
        for raw_record in dataset.take(1):
            parsed = parser.parse(raw_record)
            assert parsed.shape == (128, 128, 1)
            assert parsed.dtype == tf.float32
    
    def test_labeled_parser_parse(self, mock_labeled_tfrecord):
        """Test parsing labeled TFRecord."""
        parser = LabeledImageParser(image_size=128, channels=1)
        dataset = tf.data.TFRecordDataset(str(mock_labeled_tfrecord))
        
        for raw_record in dataset.take(1):
            parsed = parser.parse(raw_record)
            assert 'image' in parsed
            assert 'label' in parsed
            assert parsed['image'].shape == (128, 128, 1)


class TestParserFactory:
    """Tests for ParserFactory."""
    
    def test_create_unlabeled_parser(self):
        """Test creating unlabeled parser via factory."""
        parser = ParserFactory.create_parser('unlabeled', image_size=64)
        assert isinstance(parser, UnlabeledImageParser)
        assert parser.image_size == 64
    
    def test_create_labeled_parser(self):
        """Test creating labeled parser via factory."""
        parser = ParserFactory.create_parser('labeled', image_size=128)
        assert isinstance(parser, LabeledImageParser)
        assert parser.image_size == 128
    
    def test_create_flexible_parser(self):
        """Test creating flexible parser via factory."""
        parser = ParserFactory.create_parser('flexible', image_size=128)
        assert isinstance(parser, FlexibleImageParser)
    
    def test_unknown_parser_type(self):
        """Test error with unknown parser type."""
        with pytest.raises(ValueError, match="Unknown parser type"):
            ParserFactory.create_parser('invalid_type')
    
    def test_register_custom_parser(self):
        """Test registering a custom parser."""
        class CustomParser(UnlabeledImageParser):
            pass
        
        ParserFactory.register_parser('custom', CustomParser)
        parser = ParserFactory.create_parser('custom', image_size=64)
        assert isinstance(parser, CustomParser)


class TestDatasetFactory:
    """Tests for DatasetFactory."""
    
    def test_factory_creation(self):
        """Test creating dataset factory."""
        config = DataConfig(batch_size=16, image_size=128)
        factory = DatasetFactory(config)
        assert factory.config.batch_size == 16
    
    def test_create_dataset(self, mock_tfrecord):
        """Test creating dataset from TFRecord."""
        config = DataConfig(batch_size=4, image_size=128)
        factory = DatasetFactory(config)
        
        dataset = factory.create_dataset([str(mock_tfrecord)], repeat=False)
        
        # Check dataset produces batches
        for batch in dataset.take(1):
            assert batch.shape[0] == 4  # batch size
            assert batch.shape[1:] == (128, 128, 1)
    
    def test_create_training_dataset(self, mock_tfrecord):
        """Test creating training dataset (shuffled, repeated)."""
        config = DataConfig(batch_size=4)
        factory = DatasetFactory(config)
        
        dataset = factory.create_training_dataset([str(mock_tfrecord)])
        
        # Training dataset should produce batches indefinitely
        count = 0
        for batch in dataset.take(5):
            assert batch.shape[0] == 4
            count += 1
        
        assert count == 5  # Should produce 5 batches due to repeat
    
    def test_create_evaluation_dataset(self, mock_tfrecord):
        """Test creating evaluation dataset (no shuffle, no repeat)."""
        config = DataConfig(batch_size=4)
        factory = DatasetFactory(config)
        
        dataset = factory.create_evaluation_dataset([str(mock_tfrecord)])
        
        # Count total batches (should be finite)
        count = sum(1 for _ in dataset)
        assert count > 0  # At least one batch
        assert count < 100  # Not infinite
    
    def test_get_dataset_info(self, mock_tfrecord):
        """Test getting dataset information."""
        config = DataConfig()
        factory = DatasetFactory(config)
        
        info = factory.get_dataset_info([str(mock_tfrecord)])
        
        assert 'num_records' in info
        assert 'num_files' in info
        assert 'total_size_mb' in info
        assert info['num_records'] == 10  # From fixture
    
    def test_file_not_found(self):
        """Test error when TFRecord file doesn't exist."""
        config = DataConfig()
        factory = DatasetFactory(config)
        
        with pytest.raises(FileNotFoundError):
            factory.create_dataset(['nonexistent.tfrecord'])


class TestPreprocessingSteps:
    """Tests for preprocessing pipeline steps."""
    
    def test_clahe_step(self):
        """Test CLAHE preprocessing step."""
        step = CLAHEStep(clip_limit=2.0, tile_grid_size=(8, 8))
        
        # Create test image
        image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        result = step.process(image)
        
        assert result.shape == image.shape
        assert result.dtype == np.uint8
    
    def test_gamma_correction_step(self):
        """Test gamma correction step."""
        step = GammaCorrectionStep(gamma=1.5)
        
        image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        result = step.process(image)
        
        assert result.shape == image.shape
    
    def test_pixel_binning_step(self):
        """Test pixel binning step."""
        step = PixelBinningStep(bin_size=16)
        
        image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        result = step.process(image)
        
        assert result.shape == image.shape
        # Check that values are binned
        unique_values = np.unique(result)
        assert len(unique_values) <= 16
    
    def test_normalization_step(self):
        """Test normalization step."""
        step = NormalizationStep(output_range=(-1.0, 1.0))
        
        image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        result = step.process(image)
        
        assert result.dtype == np.float32
        assert result.min() >= -1.0
        assert result.max() <= 1.0
    
    def test_chain_of_responsibility(self):
        """Test chaining preprocessing steps."""
        # Create chain: CLAHE -> Gamma -> Normalization
        pipeline = CLAHEStep(
            clip_limit=2.0,
            next_step=GammaCorrectionStep(
                gamma=1.5,
                next_step=NormalizationStep(output_range=(-1.0, 1.0))
            )
        )
        
        image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        result = pipeline(image)
        
        # Final output should be normalized float32
        assert result.dtype == np.float32
        assert -1.0 <= result.min() <= result.max() <= 1.0


class TestPreprocessingPipeline:
    """Tests for preprocessing pipeline presets."""
    
    def test_medical_imaging_pipeline(self):
        """Test medical imaging preprocessing pipeline."""
        pipeline = PreprocessingPipeline.for_medical_imaging()
        
        image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        result = pipeline(image)
        
        assert result.dtype == np.float32
        assert result.shape == (128, 128)
    
    def test_simple_normalization_pipeline(self):
        """Test simple normalization pipeline."""
        pipeline = PreprocessingPipeline.for_simple_normalization(
            output_range=(0.0, 1.0)
        )
        
        image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        result = pipeline(image)
        
        assert 0.0 <= result.min() <= result.max() <= 1.0
    
    def test_retinopathy_pipeline(self):
        """Test retinopathy-specific pipeline."""
        pipeline = PreprocessingPipeline.for_retinopathy(image_size=128)
        
        # Test with different sized image
        image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        result = pipeline(image)
        
        # Should be resized to 128x128 and normalized
        assert result.shape == (128, 128)
        assert result.dtype == np.float32


@pytest.mark.integration
class TestFederatedDataLoader:
    """Integration tests for FederatedDataLoader."""
    
    def test_loader_creation(self):
        """Test creating federated data loader."""
        config = ExperimentConfig(name="test")
        loader = FederatedDataLoader(config)
        
        assert loader.config == config
        assert loader.parser_type == 'unlabeled'
    
    def test_loader_with_labeled_parser(self):
        """Test loader with labeled parser."""
        config = ExperimentConfig(name="test")
        loader = FederatedDataLoader(config, parser_type='labeled')
        
        assert loader.parser_type == 'labeled'
    
    @pytest.mark.skip(reason="Requires actual federated data files")
    def test_load_client_datasets(self):
        """Test loading client datasets."""
        # This would require actual data files
        pass
