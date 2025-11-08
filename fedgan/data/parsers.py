"""
TFRecord parsing strategies for different data formats.

This module consolidates all TFRecord parsing logic that was previously
duplicated across 6 different files in the original codebase.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import tensorflow as tf


class TFRecordParser(ABC):
    """Abstract base class for TFRecord parsing strategies.
    
    This implements the Strategy pattern, allowing different parsing
    strategies for different TFRecord formats.
    """
    
    @abstractmethod
    def parse(self, example_proto: tf.Tensor) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        """Parse a single TFRecord example.
        
        Args:
            example_proto: Serialized TFRecord example.
        
        Returns:
            Parsed tensor(s) - either a single tensor or dict of tensors.
        """
        pass
    
    @property
    @abstractmethod
    def output_signature(self) -> Union[tf.TensorSpec, Dict[str, tf.TensorSpec]]:
        """Get the output signature for this parser.
        
        Returns:
            TensorSpec or dict of TensorSpecs describing the output.
        """
        pass


class UnlabeledImageParser(TFRecordParser):
    """Parser for unlabeled image TFRecords.
    
    Expected format: Single serialized image tensor.
    
    Args:
        image_size: Size of the image (assumed square).
        channels: Number of image channels (1 for grayscale, 3 for RGB).
    """
    
    def __init__(self, image_size: int = 128, channels: int = 1):
        self.image_size = image_size
        self.channels = channels
    
    def parse(self, example_proto: tf.Tensor) -> tf.Tensor:
        """Parse unlabeled image from TFRecord.
        
        Args:
            example_proto: Serialized example.
        
        Returns:
            Image tensor of shape (image_size, image_size, channels).
        """
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
        }
        
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.io.decode_raw(parsed['image'], tf.float32)
        image = tf.reshape(image, [self.image_size, self.image_size, self.channels])
        
        return image
    
    @property
    def output_signature(self) -> tf.TensorSpec:
        """Output signature for unlabeled images."""
        return tf.TensorSpec(
            shape=(self.image_size, self.image_size, self.channels),
            dtype=tf.float32
        )


class LabeledImageParser(TFRecordParser):
    """Parser for labeled image TFRecords.
    
    Expected format: Image and integer label.
    
    Args:
        image_size: Size of the image (assumed square).
        channels: Number of image channels.
        num_classes: Number of classes (for validation).
    """
    
    def __init__(self, image_size: int = 128, channels: int = 1, num_classes: int = 5):
        self.image_size = image_size
        self.channels = channels
        self.num_classes = num_classes
    
    def parse(self, example_proto: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Parse labeled image from TFRecord.
        
        Args:
            example_proto: Serialized example.
        
        Returns:
            Dictionary with 'image' and 'label' tensors.
        """
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.io.decode_raw(parsed['image'], tf.float32)
        image = tf.reshape(image, [self.image_size, self.image_size, self.channels])
        label = tf.cast(parsed['label'], tf.int32)
        
        return {'image': image, 'label': label}
    
    @property
    def output_signature(self) -> Dict[str, tf.TensorSpec]:
        """Output signature for labeled images."""
        return {
            'image': tf.TensorSpec(
                shape=(self.image_size, self.image_size, self.channels),
                dtype=tf.float32
            ),
            'label': tf.TensorSpec(shape=(), dtype=tf.int32)
        }


class FlexibleImageParser(TFRecordParser):
    """Flexible parser that handles both labeled and unlabeled formats.
    
    This parser tries labeled format first, falls back to unlabeled.
    Useful for datasets with mixed formats.
    
    Args:
        image_size: Size of the image.
        channels: Number of image channels.
    """
    
    def __init__(self, image_size: int = 128, channels: int = 1):
        self.image_size = image_size
        self.channels = channels
        self._labeled_parser = LabeledImageParser(image_size, channels)
        self._unlabeled_parser = UnlabeledImageParser(image_size, channels)
    
    def parse(self, example_proto: tf.Tensor) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        """Parse image, trying labeled format first.
        
        Args:
            example_proto: Serialized example.
        
        Returns:
            Either a dict with image and label, or just image tensor.
        """
        # Try labeled format first
        try:
            return self._labeled_parser.parse(example_proto)
        except tf.errors.InvalidArgumentError:
            # Fall back to unlabeled
            return self._unlabeled_parser.parse(example_proto)
    
    @property
    def output_signature(self) -> tf.TensorSpec:
        """Output signature (unlabeled format for simplicity)."""
        return self._unlabeled_parser.output_signature


class ParserFactory:
    """Factory for creating TFRecord parsers.
    
    This provides a central registry of available parsers.
    """
    
    _PARSERS = {
        'unlabeled': UnlabeledImageParser,
        'labeled': LabeledImageParser,
        'flexible': FlexibleImageParser,
    }
    
    @classmethod
    def create_parser(
        cls,
        parser_type: str,
        image_size: int = 128,
        channels: int = 1,
        **kwargs: Any
    ) -> TFRecordParser:
        """Create a TFRecord parser by type.
        
        Args:
            parser_type: Type of parser ('unlabeled', 'labeled', 'flexible').
            image_size: Size of images.
            channels: Number of channels.
            **kwargs: Additional parser-specific arguments.
        
        Returns:
            Configured TFRecordParser instance.
        
        Raises:
            ValueError: If parser_type is unknown.
        """
        parser_class = cls._PARSERS.get(parser_type)
        if parser_class is None:
            raise ValueError(
                f"Unknown parser type: {parser_type}. "
                f"Available: {list(cls._PARSERS.keys())}"
            )
        
        return parser_class(image_size=image_size, channels=channels, **kwargs)
    
    @classmethod
    def register_parser(cls, name: str, parser_class: type) -> None:
        """Register a custom parser type.
        
        Args:
            name: Name for the parser type.
            parser_class: Parser class (must inherit from TFRecordParser).
        """
        if not issubclass(parser_class, TFRecordParser):
            raise TypeError("Parser class must inherit from TFRecordParser")
        cls._PARSERS[name] = parser_class
