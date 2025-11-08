"""Data loading and preprocessing for FedGAN."""

from fedgan.data.factory import DatasetFactory, MultiDatasetFactory
from fedgan.data.loaders import DataLoaderBuilder, FederatedDataLoader
from fedgan.data.parsers import (
    FlexibleImageParser,
    LabeledImageParser,
    ParserFactory,
    TFRecordParser,
    UnlabeledImageParser,
)
from fedgan.data.preprocessing import (
    CLAHEStep,
    GammaCorrectionStep,
    NormalizationStep,
    PixelBinningStep,
    PreprocessingPipeline,
    PreprocessingStep,
    ResizeStep,
)

__all__ = [
    # Parsers
    "TFRecordParser",
    "UnlabeledImageParser",
    "LabeledImageParser",
    "FlexibleImageParser",
    "ParserFactory",
    # Factories
    "DatasetFactory",
    "MultiDatasetFactory",
    # Loaders
    "FederatedDataLoader",
    "DataLoaderBuilder",
    # Preprocessing
    "PreprocessingStep",
    "CLAHEStep",
    "GammaCorrectionStep",
    "PixelBinningStep",
    "NormalizationStep",
    "ResizeStep",
    "PreprocessingPipeline",
]
