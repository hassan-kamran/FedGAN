"""
Factory for creating TensorFlow datasets from TFRecords.

This module provides a unified interface for dataset creation,
eliminating the duplication that existed across multiple files.
"""
from pathlib import Path
from typing import List, Optional, Union

import tensorflow as tf

from fedgan.config import DataConfig
from fedgan.data.parsers import ParserFactory, TFRecordParser


class DatasetFactory:
    """Factory for creating configured TensorFlow datasets.
    
    This class implements the Factory pattern to create datasets with
    consistent configuration (batching, shuffling, prefetching, etc.).
    
    Args:
        config: Data configuration object.
        parser: TFRecord parser to use (or None to use default).
    
    Example:
        >>> from fedgan.config import DataConfig
        >>> config = DataConfig(batch_size=16, image_size=128)
        >>> factory = DatasetFactory(config)
        >>> dataset = factory.create_dataset(['path/to/data.tfrecord'])
    """
    
    def __init__(
        self,
        config: DataConfig,
        parser: Optional[TFRecordParser] = None
    ):
        self.config = config
        self.parser = parser or ParserFactory.create_parser(
            'unlabeled',
            image_size=config.image_size,
            channels=config.channels
        )
    
    def create_dataset(
        self,
        tfrecord_paths: Union[str, List[str]],
        shuffle: bool = True,
        repeat: bool = True,
        batch_size: Optional[int] = None,
        compression_type: str = 'GZIP',
        num_parallel_calls: int = tf.data.AUTOTUNE,
    ) -> tf.data.Dataset:
        """Create a TensorFlow dataset from TFRecord files.
        
        Args:
            tfrecord_paths: Path(s) to TFRecord file(s).
            shuffle: Whether to shuffle the dataset.
            repeat: Whether to repeat the dataset indefinitely.
            batch_size: Batch size (uses config.batch_size if None).
            compression_type: TFRecord compression ('GZIP', 'ZLIB', or '').
            num_parallel_calls: Parallelism for map operations.
        
        Returns:
            Configured tf.data.Dataset ready for training.
        """
        # Ensure paths is a list
        if isinstance(tfrecord_paths, (str, Path)):
            tfrecord_paths = [str(tfrecord_paths)]
        else:
            tfrecord_paths = [str(p) for p in tfrecord_paths]
        
        # Validate files exist
        for path in tfrecord_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"TFRecord not found: {path}")
        
        # Create base dataset
        dataset = tf.data.TFRecordDataset(
            tfrecord_paths,
            compression_type=compression_type,
            num_parallel_reads=num_parallel_calls
        )
        
        # Parse records
        dataset = dataset.map(
            self.parser.parse,
            num_parallel_calls=num_parallel_calls
        )
        
        # Cache dataset in memory if small enough
        dataset = dataset.cache()
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=self.config.shuffle_buffer_size,
                reshuffle_each_iteration=True
            )
        
        # Repeat if requested
        if repeat:
            dataset = dataset.repeat()
        
        # Batch
        batch_size = batch_size or self.config.batch_size
        dataset = dataset.batch(batch_size, drop_remainder=True)
        
        # Prefetch for performance
        dataset = dataset.prefetch(buffer_size=self.config.prefetch_size)
        
        return dataset
    
    def create_training_dataset(
        self,
        tfrecord_paths: Union[str, List[str]],
        **kwargs
    ) -> tf.data.Dataset:
        """Create a dataset optimized for training.
        
        Args:
            tfrecord_paths: Path(s) to TFRecord file(s).
            **kwargs: Additional arguments for create_dataset.
        
        Returns:
            Dataset configured for training (shuffled, repeated, batched).
        """
        return self.create_dataset(
            tfrecord_paths,
            shuffle=True,
            repeat=True,
            **kwargs
        )
    
    def create_evaluation_dataset(
        self,
        tfrecord_paths: Union[str, List[str]],
        **kwargs
    ) -> tf.data.Dataset:
        """Create a dataset optimized for evaluation.
        
        Args:
            tfrecord_paths: Path(s) to TFRecord file(s).
            **kwargs: Additional arguments for create_dataset.
        
        Returns:
            Dataset configured for evaluation (no shuffle, no repeat).
        """
        return self.create_dataset(
            tfrecord_paths,
            shuffle=False,
            repeat=False,
            **kwargs
        )
    
    def get_dataset_info(
        self,
        tfrecord_paths: Union[str, List[str]]
    ) -> dict:
        """Get information about a TFRecord dataset.
        
        Args:
            tfrecord_paths: Path(s) to TFRecord file(s).
        
        Returns:
            Dictionary with dataset statistics.
        """
        if isinstance(tfrecord_paths, (str, Path)):
            tfrecord_paths = [str(tfrecord_paths)]
        
        # Count total records
        dataset = tf.data.TFRecordDataset(tfrecord_paths)
        num_records = sum(1 for _ in dataset)
        
        # Get file sizes
        total_size = sum(Path(p).stat().st_size for p in tfrecord_paths)
        
        return {
            'num_records': num_records,
            'num_files': len(tfrecord_paths),
            'total_size_mb': total_size / (1024 * 1024),
            'avg_size_per_record': total_size / num_records if num_records > 0 else 0,
        }


class MultiDatasetFactory(DatasetFactory):
    """Factory for creating datasets from multiple TFRecord sources.
    
    This is useful for combining data from different clients or sources.
    
    Args:
        config: Data configuration.
        parser: TFRecord parser.
    """
    
    def create_interleaved_dataset(
        self,
        tfrecord_path_lists: List[List[str]],
        cycle_length: Optional[int] = None,
        **kwargs
    ) -> tf.data.Dataset:
        """Create a dataset that interleaves multiple sources.
        
        Args:
            tfrecord_path_lists: List of lists of TFRecord paths.
            cycle_length: Number of datasets to interleave at once.
            **kwargs: Additional arguments for create_dataset.
        
        Returns:
            Interleaved dataset from all sources.
        """
        cycle_length = cycle_length or len(tfrecord_path_lists)
        
        # Create a dataset of datasets
        def create_single_dataset(paths):
            return self.create_dataset(
                paths.numpy().decode('utf-8'),
                **kwargs
            )
        
        # Convert paths to dataset
        paths_dataset = tf.data.Dataset.from_tensor_slices(
            [str(p) for paths in tfrecord_path_lists for p in paths]
        )
        
        # Interleave
        dataset = paths_dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x),
            cycle_length=cycle_length,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Parse and configure
        dataset = dataset.map(
            self.parser.parse,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if kwargs.get('shuffle', True):
            dataset = dataset.shuffle(self.config.shuffle_buffer_size)
        
        if kwargs.get('repeat', True):
            dataset = dataset.repeat()
        
        batch_size = kwargs.get('batch_size', self.config.batch_size)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self.config.prefetch_size)
        
        return dataset
    
    def create_concatenated_dataset(
        self,
        tfrecord_path_lists: List[List[str]],
        **kwargs
    ) -> tf.data.Dataset:
        """Create a dataset by concatenating multiple sources.
        
        Args:
            tfrecord_path_lists: List of lists of TFRecord paths.
            **kwargs: Additional arguments for create_dataset.
        
        Returns:
            Concatenated dataset from all sources.
        """
        datasets = [
            self.create_dataset(paths, **kwargs)
            for paths in tfrecord_path_lists
        ]
        
        # Concatenate all datasets
        combined = datasets[0]
        for dataset in datasets[1:]:
            combined = combined.concatenate(dataset)
        
        return combined
