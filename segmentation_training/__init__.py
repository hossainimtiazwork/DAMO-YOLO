"""
Segmentation Training Pipeline Module

This module provides a complete training pipeline for semantic segmentation using:
- SMP (Segmentation Models PyTorch) UNet architecture
- PyTorch Lightning training framework
- Albumentations for data augmentation
- TensorBoard for logging and visualization

Components:
-----------
- train_unet.py: Main training script with data module and model
- inference.py: Inference script for trained models
- config.yaml: Configuration file for training parameters
- example_usage.py: Example demonstrating the pipeline

Usage:
------
See README.md for detailed usage instructions.
"""

from .train_unet import (
    SegmentationDataset,
    SegmentationDataModule,
    SegmentationModel,
    train,
)

__version__ = '1.0.0'
__all__ = [
    'SegmentationDataset',
    'SegmentationDataModule', 
    'SegmentationModel',
    'train',
]
