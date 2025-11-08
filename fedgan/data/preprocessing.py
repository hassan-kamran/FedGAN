"""
Image preprocessing pipeline using Chain of Responsibility pattern.

This module consolidates preprocessing logic from the original
preprocessing.py and provides a composable pipeline approach.
"""
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np


class PreprocessingStep(ABC):
    """Abstract base for preprocessing steps (Chain of Responsibility).
    
    Each preprocessing step can be chained with the next step,
    allowing for flexible, composable preprocessing pipelines.
    
    Args:
        next_step: Next preprocessing step in the chain.
    """
    
    def __init__(self, next_step: Optional['PreprocessingStep'] = None):
        self._next = next_step
    
    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """Process a single image.
        
        Args:
            image: Input image as numpy array.
        
        Returns:
            Processed image.
        """
        pass
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply this step and all subsequent steps.
        
        Args:
            image: Input image.
        
        Returns:
            Fully processed image.
        """
        # Apply this step
        image = self.process(image)
        
        # Apply next step if exists
        if self._next is not None:
            image = self._next(image)
        
        return image
    
    def add_step(self, step: 'PreprocessingStep') -> 'PreprocessingStep':
        """Add a step to the end of the chain.
        
        Args:
            step: Preprocessing step to add.
        
        Returns:
            The added step (for chaining).
        """
        if self._next is None:
            self._next = step
        else:
            self._next.add_step(step)
        return step


class CLAHEStep(PreprocessingStep):
    """Contrast Limited Adaptive Histogram Equalization.
    
    Args:
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.
        next_step: Next step in the pipeline.
    """
    
    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple = (8, 8),
        next_step: Optional[PreprocessingStep] = None
    ):
        super().__init__(next_step)
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to image.
        
        Args:
            image: Input image (uint8).
        
        Returns:
            CLAHE-enhanced image.
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Apply CLAHE
        return self.clahe.apply(image)


class GammaCorrectionStep(PreprocessingStep):
    """Gamma correction for brightness adjustment.
    
    Args:
        gamma: Gamma value (>1 brightens, <1 darkens).
        next_step: Next step in the pipeline.
    """
    
    def __init__(
        self,
        gamma: float = 1.5,
        next_step: Optional[PreprocessingStep] = None
    ):
        super().__init__(next_step)
        self.gamma = gamma
        
        # Precompute lookup table for efficiency
        inv_gamma = 1.0 / gamma
        self.lookup_table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype(np.uint8)
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply gamma correction.
        
        Args:
            image: Input image (uint8).
        
        Returns:
            Gamma-corrected image.
        """
        # Ensure uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Apply lookup table
        return cv2.LUT(image, self.lookup_table)


class PixelBinningStep(PreprocessingStep):
    """Pixel intensity binning/quantization.
    
    Args:
        bin_size: Number of intensity bins (e.g., 16 for 16 levels).
        next_step: Next step in the pipeline.
    """
    
    def __init__(
        self,
        bin_size: int = 16,
        next_step: Optional[PreprocessingStep] = None
    ):
        super().__init__(next_step)
        self.bin_size = bin_size
        self.bin_width = 256 // bin_size
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply pixel binning.
        
        Args:
            image: Input image (uint8).
        
        Returns:
            Binned image.
        """
        # Ensure uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Bin pixels
        binned = (image // self.bin_width) * self.bin_width
        return binned.astype(np.uint8)


class NormalizationStep(PreprocessingStep):
    """Normalize pixel values to specified range.
    
    Args:
        output_range: Tuple of (min, max) for output range.
        input_range: Tuple of (min, max) for input range.
        next_step: Next step in the pipeline.
    """
    
    def __init__(
        self,
        output_range: tuple = (-1.0, 1.0),
        input_range: tuple = (0, 255),
        next_step: Optional[PreprocessingStep] = None
    ):
        super().__init__(next_step)
        self.output_min, self.output_max = output_range
        self.input_min, self.input_max = input_range
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to target range.
        
        Args:
            image: Input image.
        
        Returns:
            Normalized image as float32.
        """
        # Convert to float
        image = image.astype(np.float32)
        
        # Normalize from input range to [0, 1]
        image = (image - self.input_min) / (self.input_max - self.input_min)
        
        # Scale to output range
        image = image * (self.output_max - self.output_min) + self.output_min
        
        return image


class ResizeStep(PreprocessingStep):
    """Resize image to target size.
    
    Args:
        target_size: Target size as (height, width).
        interpolation: OpenCV interpolation method.
        next_step: Next step in the pipeline.
    """
    
    def __init__(
        self,
        target_size: tuple = (128, 128),
        interpolation: int = cv2.INTER_LINEAR,
        next_step: Optional[PreprocessingStep] = None
    ):
        super().__init__(next_step)
        self.target_size = target_size
        self.interpolation = interpolation
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Resize image.
        
        Args:
            image: Input image.
        
        Returns:
            Resized image.
        """
        return cv2.resize(
            image,
            self.target_size,
            interpolation=self.interpolation
        )


class ReshapeStep(PreprocessingStep):
    """Reshape image array (e.g., add channel dimension).
    
    Args:
        target_shape: Target shape for the image.
        next_step: Next step in the pipeline.
    """
    
    def __init__(
        self,
        target_shape: tuple,
        next_step: Optional[PreprocessingStep] = None
    ):
        super().__init__(next_step)
        self.target_shape = target_shape
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Reshape image array.
        
        Args:
            image: Input image.
        
        Returns:
            Reshaped image.
        """
        return image.reshape(self.target_shape)


class PreprocessingPipeline:
    """Convenience class for building preprocessing pipelines.
    
    Example:
        >>> from fedgan.data.preprocessing import PreprocessingPipeline
        >>> pipeline = PreprocessingPipeline.for_medical_imaging()
        >>> processed_image = pipeline(raw_image)
    """
    
    @staticmethod
    def for_medical_imaging(
        clip_limit: float = 2.0,
        gamma: float = 1.5,
        bin_size: int = 16,
        output_range: tuple = (-1.0, 1.0)
    ) -> PreprocessingStep:
        """Create standard medical imaging preprocessing pipeline.
        
        Pipeline: CLAHE -> Gamma Correction -> Binning -> Normalization
        
        Args:
            clip_limit: CLAHE clip limit.
            gamma: Gamma correction value.
            bin_size: Number of intensity bins.
            output_range: Output normalization range.
        
        Returns:
            Configured preprocessing pipeline.
        """
        return CLAHEStep(
            clip_limit=clip_limit,
            tile_grid_size=(8, 8),
            next_step=GammaCorrectionStep(
                gamma=gamma,
                next_step=PixelBinningStep(
                    bin_size=bin_size,
                    next_step=NormalizationStep(output_range=output_range)
                )
            )
        )
    
    @staticmethod
    def for_simple_normalization(
        output_range: tuple = (-1.0, 1.0)
    ) -> PreprocessingStep:
        """Create simple normalization pipeline.
        
        Pipeline: Normalization only
        
        Args:
            output_range: Output normalization range.
        
        Returns:
            Normalization step.
        """
        return NormalizationStep(output_range=output_range)
    
    @staticmethod
    def for_retinopathy(
        image_size: int = 128
    ) -> PreprocessingStep:
        """Create preprocessing pipeline for diabetic retinopathy images.
        
        This replicates the original preprocessing from the PLOS ONE paper.
        
        Args:
            image_size: Target image size.
        
        Returns:
            Configured preprocessing pipeline.
        """
        return ResizeStep(
            target_size=(image_size, image_size),
            next_step=CLAHEStep(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                next_step=GammaCorrectionStep(
                    gamma=1.5,
                    next_step=PixelBinningStep(
                        bin_size=16,
                        next_step=NormalizationStep(
                            output_range=(-1.0, 1.0)
                        )
                    )
                )
            )
        )
