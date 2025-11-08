"""
Pytest configuration and shared fixtures for FedGAN tests.
"""
import os
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory) -> Path:
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def sample_image() -> np.ndarray:
    """Generate a sample grayscale image."""
    return np.random.randn(128, 128, 1).astype(np.float32)


@pytest.fixture
def sample_batch() -> np.ndarray:
    """Generate a batch of sample images."""
    return np.random.randn(16, 128, 128, 1).astype(np.float32)


@pytest.fixture
def sample_latent_vector() -> np.ndarray:
    """Generate a sample latent vector."""
    return np.random.randn(200).astype(np.float32)


@pytest.fixture
def sample_latent_batch() -> np.ndarray:
    """Generate a batch of latent vectors."""
    return np.random.randn(16, 200).astype(np.float32)


@pytest.fixture
def mock_tfrecord(tmp_path: Path) -> Path:
    """Create a mock TFRecord file with sample data."""
    tfrecord_path = tmp_path / "test.tfrecord"

    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        for i in range(10):
            image = np.random.randn(128, 128, 1).astype(np.float32)
            feature = {
                "image": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[image.tobytes()])
                ),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    return tfrecord_path


@pytest.fixture
def mock_labeled_tfrecord(tmp_path: Path) -> Path:
    """Create a mock labeled TFRecord file."""
    tfrecord_path = tmp_path / "test_labeled.tfrecord"

    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        for i in range(10):
            image = np.random.randn(128, 128, 1).astype(np.float32)
            label = i % 5  # 5 classes
            feature = {
                "image": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[image.tobytes()])
                ),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    return tfrecord_path


@pytest.fixture
def disable_gpu() -> Generator[None, None, None]:
    """Disable GPU for CPU-only tests."""
    original_devices = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices([], "GPU")
    yield
    # Restore GPU visibility
    if original_devices:
        tf.config.set_visible_devices(original_devices, "GPU")


# Add markers for different test types
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "e2e: mark test as an end-to-end test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
