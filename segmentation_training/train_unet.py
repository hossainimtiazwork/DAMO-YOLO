#!/usr/bin/env python3
"""
Comprehensive Training Pipeline for Segmentation with:
- SMP (Segmentation Models PyTorch) UNet
- PyTorch Lightning for training framework
- Albumentations for data augmentation
- Configurable optimizer and scheduler
- TensorBoard for logging and visualization

This script provides a complete, production-ready training pipeline
for semantic segmentation tasks.
"""

import os
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2


# ============================================================================
# Data Module: Handles data loading, augmentation, and preparation
# ============================================================================

class SegmentationDataset(Dataset):
    """
    Custom Dataset for semantic segmentation tasks.
    
    This dataset handles loading images and their corresponding masks,
    applying augmentations, and preparing data for training.
    
    Args:
        images_dir (str): Path to directory containing images
        masks_dir (str): Path to directory containing segmentation masks
        transform (albumentations.Compose): Augmentation pipeline
        image_size (tuple): Target size for images (height, width)
    """
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (256, 256),
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_size = image_size
        
        # Get list of image filenames (assuming images and masks have same names)
        self.image_filenames = sorted(os.listdir(images_dir))
        
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.image_filenames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a single sample (image, mask pair).
        
        Args:
            idx (int): Index of the sample to load
            
        Returns:
            tuple: (image, mask) where both are torch.Tensors
        """
        # Load image
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load corresponding mask
        # Assumes mask has same name but different extension (e.g., .png)
        # Use splitext for robust extension handling
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply augmentations if provided
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert mask to long tensor for CrossEntropyLoss
        mask = torch.from_numpy(mask).long()
        
        return image, mask


class SegmentationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for segmentation tasks.
    
    This module handles:
    - Data loading and splitting
    - Augmentation pipelines (different for train/val)
    - DataLoader creation with appropriate settings
    
    Args:
        train_images_dir (str): Path to training images
        train_masks_dir (str): Path to training masks
        val_images_dir (str): Path to validation images
        val_masks_dir (str): Path to validation masks
        batch_size (int): Batch size for training and validation
        num_workers (int): Number of workers for data loading
        image_size (tuple): Target image size (height, width)
    """
    
    def __init__(
        self,
        train_images_dir: str,
        train_masks_dir: str,
        val_images_dir: str,
        val_masks_dir: str,
        batch_size: int = 8,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (256, 256),
    ):
        super().__init__()
        self.train_images_dir = train_images_dir
        self.train_masks_dir = train_masks_dir
        self.val_images_dir = val_images_dir
        self.val_masks_dir = val_masks_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
    def get_train_transform(self) -> A.Compose:
        """
        Create augmentation pipeline for training data.
        
        Includes various augmentations to improve model robustness:
        - Geometric: Resize, Random rotations, flips, shifts
        - Color: Brightness, contrast, saturation adjustments
        - Normalization: ImageNet statistics
        
        Returns:
            albumentations.Compose: Training augmentation pipeline
        """
        return A.Compose([
            # Resize to target size
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),  # Flip horizontally with 50% probability
            A.VerticalFlip(p=0.5),     # Flip vertically with 50% probability
            # Combined shift, scale, and rotation augmentation
            A.ShiftScaleRotate(
                shift_limit=0.1,        # Max shift fraction
                scale_limit=0.2,        # Max scale change
                rotate_limit=35,        # Max rotation degrees
                p=0.5
            ),
            
            # Color augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.2,   # Brightness adjustment range
                contrast_limit=0.2,     # Contrast adjustment range
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,     # Hue adjustment
                sat_shift_limit=30,     # Saturation adjustment
                val_shift_limit=20,     # Value adjustment
                p=0.5
            ),
            
            # Blur and noise
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # Add gaussian blur
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # Add gaussian noise
            
            # Normalization using ImageNet statistics
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225],   # ImageNet std
                max_pixel_value=255.0,
            ),
            
            # Convert to PyTorch tensor
            ToTensorV2(),
        ])
    
    def get_val_transform(self) -> A.Compose:
        """
        Create augmentation pipeline for validation data.
        
        Only includes essential transformations (resize and normalize)
        without random augmentations for consistent validation.
        
        Returns:
            albumentations.Compose: Validation augmentation pipeline
        """
        return A.Compose([
            # Resize to target size
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            
            # Normalization using ImageNet statistics
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            
            # Convert to PyTorch tensor
            ToTensorV2(),
        ])
    
    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for training and validation.
        
        This is called by PyTorch Lightning before training/validation begins.
        
        Args:
            stage (str, optional): Either 'fit', 'validate', 'test', or 'predict'
        """
        if stage == 'fit' or stage is None:
            # Create training dataset with augmentations
            self.train_dataset = SegmentationDataset(
                images_dir=self.train_images_dir,
                masks_dir=self.train_masks_dir,
                transform=self.get_train_transform(),
                image_size=self.image_size,
            )
            
            # Create validation dataset without augmentations
            self.val_dataset = SegmentationDataset(
                images_dir=self.val_images_dir,
                masks_dir=self.val_masks_dir,
                transform=self.get_val_transform(),
                image_size=self.image_size,
            )
    
    def train_dataloader(self) -> DataLoader:
        """
        Create DataLoader for training data.
        
        Returns:
            DataLoader: Training data loader with shuffling enabled
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle for better training
            num_workers=self.num_workers,
            pin_memory=True,  # Speed up GPU transfer
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def val_dataloader(self) -> DataLoader:
        """
        Create DataLoader for validation data.
        
        Returns:
            DataLoader: Validation data loader without shuffling
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffling for validation
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )


# ============================================================================
# Model Module: Defines the segmentation model architecture
# ============================================================================

class SegmentationModel(pl.LightningModule):
    """
    PyTorch Lightning Module for semantic segmentation using SMP UNet.
    
    This module encapsulates:
    - Model architecture (UNet from segmentation_models_pytorch)
    - Loss function (CrossEntropyLoss for multi-class segmentation)
    - Optimizer configuration (AdamW with weight decay)
    - Learning rate scheduler (CosineAnnealingLR)
    - Training and validation step logic
    - Metrics computation (IoU, accuracy)
    
    Args:
        encoder_name (str): Backbone encoder (e.g., 'resnet34', 'efficientnet-b0')
        encoder_weights (str): Pre-trained weights ('imagenet', None)
        in_channels (int): Number of input channels (3 for RGB)
        num_classes (int): Number of segmentation classes
        learning_rate (float): Initial learning rate
        weight_decay (float): Weight decay for optimizer
        max_epochs (int): Maximum training epochs for scheduler
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet34',
        encoder_weights: Optional[str] = 'imagenet',
        in_channels: int = 3,
        num_classes: int = 2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
    ):
        super().__init__()
        
        # Save hyperparameters for logging and checkpointing
        self.save_hyperparameters()
        
        # Initialize UNet model from segmentation_models_pytorch
        # UNet is a popular architecture for segmentation with encoder-decoder structure
        self.model = smp.Unet(
            encoder_name=encoder_name,        # Backbone encoder
            encoder_weights=encoder_weights,  # Use pre-trained weights
            in_channels=in_channels,          # Input channels (RGB = 3)
            classes=num_classes,              # Number of output classes
            activation=None,                  # No activation (we'll use softmax in loss)
        )
        
        # Define loss function
        # CrossEntropyLoss combines LogSoftmax and NLLLoss
        # It's suitable for multi-class segmentation
        self.criterion = nn.CrossEntropyLoss()
        
        # Store training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input images [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Segmentation logits [batch_size, num_classes, height, width]
        """
        return self.model(x)
    
    def compute_metrics(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Compute evaluation metrics (IoU and pixel accuracy).
        
        Args:
            preds (torch.Tensor): Predicted logits [batch_size, num_classes, H, W]
            targets (torch.Tensor): Ground truth masks [batch_size, H, W]
            
        Returns:
            tuple: (mean_iou, pixel_accuracy)
        """
        # Convert logits to class predictions
        preds = torch.argmax(preds, dim=1)  # [batch_size, H, W]
        
        # Compute pixel accuracy (percentage of correct predictions)
        correct = (preds == targets).float().sum()
        total = targets.numel()
        pixel_acc = correct / total
        
        # Compute IoU (Intersection over Union) for each class
        num_classes = self.hparams.num_classes
        iou_per_class = []
        
        for cls in range(num_classes):
            # Create binary masks for current class
            pred_mask = (preds == cls)
            target_mask = (targets == cls)
            
            # Calculate intersection and union
            intersection = (pred_mask & target_mask).float().sum()
            union = (pred_mask | target_mask).float().sum()
            
            # Compute IoU (handle division by zero)
            if union > 0:
                iou = intersection / union
                iou_per_class.append(iou.item())
        
        # Compute mean IoU across all classes
        mean_iou = np.mean(iou_per_class) if iou_per_class else 0.0
        
        return mean_iou, pixel_acc.item()
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for a single batch.
        
        Args:
            batch (tuple): (images, masks) from DataLoader
            batch_idx (int): Index of the current batch
            
        Returns:
            torch.Tensor: Loss value for this batch
        """
        # Unpack batch
        images, masks = batch
        
        # Forward pass
        logits = self.forward(images)
        
        # Compute loss
        loss = self.criterion(logits, masks)
        
        # Compute metrics
        mean_iou, pixel_acc = self.compute_metrics(logits, masks)
        
        # Log metrics to TensorBoard
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/iou', mean_iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/pixel_acc', pixel_acc, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step for a single batch.
        
        Args:
            batch (tuple): (images, masks) from DataLoader
            batch_idx (int): Index of the current batch
            
        Returns:
            torch.Tensor: Loss value for this batch
        """
        # Unpack batch
        images, masks = batch
        
        # Forward pass
        logits = self.forward(images)
        
        # Compute loss
        loss = self.criterion(logits, masks)
        
        # Compute metrics
        mean_iou, pixel_acc = self.compute_metrics(logits, masks)
        
        # Log metrics to TensorBoard
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/iou', mean_iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/pixel_acc', pixel_acc, on_step=False, on_epoch=True)
        
        # Log sample predictions to TensorBoard (first batch only)
        if batch_idx == 0:
            self._log_images(images, masks, logits)
        
        return loss
    
    def _log_images(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        logits: torch.Tensor,
        num_samples: int = 4
    ):
        """
        Log sample images, ground truth masks, and predictions to TensorBoard.
        
        Args:
            images (torch.Tensor): Input images
            masks (torch.Tensor): Ground truth masks
            logits (torch.Tensor): Predicted logits
            num_samples (int): Number of samples to log
        """
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        
        # Select a subset of images to log
        num_samples = min(num_samples, images.shape[0])
        
        # Log images to TensorBoard
        for i in range(num_samples):
            # De-normalize image for visualization
            img = images[i].cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            
            # Log to TensorBoard
            self.logger.experiment.add_image(
                f'images/{i}', img, self.current_epoch
            )
            self.logger.experiment.add_image(
                f'masks_gt/{i}', masks[i].unsqueeze(0).float() / self.hparams.num_classes,
                self.current_epoch
            )
            self.logger.experiment.add_image(
                f'masks_pred/{i}', preds[i].unsqueeze(0).float() / self.hparams.num_classes,
                self.current_epoch
            )
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Uses:
        - AdamW optimizer: Adam with weight decay for better regularization
        - CosineAnnealingLR scheduler: Gradually reduces LR following cosine curve
        
        Returns:
            dict: Configuration with optimizer and scheduler
        """
        # Initialize AdamW optimizer
        # AdamW decouples weight decay from gradient updates for better performance
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),  # Default Adam beta parameters
            eps=1e-8,            # Epsilon for numerical stability
        )
        
        # Initialize Cosine Annealing scheduler
        # This gradually reduces learning rate following a cosine curve
        # Helps model converge to better local minima
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,  # Number of epochs for full cosine cycle
            eta_min=1e-6,           # Minimum learning rate
        )
        
        # Return optimizer and scheduler configuration
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # Update LR every epoch
                'frequency': 1,       # Update every epoch
                'monitor': 'val/loss',  # Metric to monitor
            }
        }


# ============================================================================
# Training Function: Main entry point for training
# ============================================================================

def train(
    train_images_dir: str,
    train_masks_dir: str,
    val_images_dir: str,
    val_masks_dir: str,
    output_dir: str = './outputs',
    encoder_name: str = 'resnet34',
    encoder_weights: str = 'imagenet',
    num_classes: int = 2,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (256, 256),
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    accelerator: str = 'auto',
    devices: int = 1,
    precision: str = '16-mixed',
):
    """
    Main training function that orchestrates the entire training pipeline.
    
    This function:
    1. Sets up the data module with augmentations
    2. Initializes the model
    3. Configures callbacks (checkpointing, early stopping, LR monitoring)
    4. Sets up TensorBoard logging
    5. Creates PyTorch Lightning trainer
    6. Runs training
    
    Args:
        train_images_dir (str): Path to training images
        train_masks_dir (str): Path to training masks
        val_images_dir (str): Path to validation images
        val_masks_dir (str): Path to validation masks
        output_dir (str): Directory to save outputs
        encoder_name (str): Encoder backbone name
        encoder_weights (str): Pre-trained weights source
        num_classes (int): Number of segmentation classes
        batch_size (int): Training batch size
        num_workers (int): Number of data loading workers
        image_size (tuple): Input image size (H, W)
        learning_rate (float): Initial learning rate
        weight_decay (float): Weight decay for regularization
        max_epochs (int): Maximum number of training epochs
        accelerator (str): Hardware accelerator ('gpu', 'cpu', 'auto')
        devices (int): Number of devices to use
        precision (str): Training precision ('32', '16-mixed', 'bf16-mixed')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================================================
    # Step 1: Initialize Data Module
    # ========================================================================
    print("=" * 80)
    print("Initializing Data Module with Albumentations...")
    print("=" * 80)
    
    data_module = SegmentationDataModule(
        train_images_dir=train_images_dir,
        train_masks_dir=train_masks_dir,
        val_images_dir=val_images_dir,
        val_masks_dir=val_masks_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
    )
    
    # ========================================================================
    # Step 2: Initialize Model
    # ========================================================================
    print("\nInitializing SMP UNet Model...")
    print("=" * 80)
    print(f"Encoder: {encoder_name}")
    print(f"Pre-trained weights: {encoder_weights}")
    print(f"Number of classes: {num_classes}")
    print("=" * 80)
    
    model = SegmentationModel(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        num_classes=num_classes,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
    )
    
    # ========================================================================
    # Step 3: Setup Callbacks
    # ========================================================================
    print("\nSetting up Training Callbacks...")
    print("=" * 80)
    
    # Checkpoint callback: Saves best models based on validation IoU
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='unet-{epoch:02d}-{val/iou:.4f}',
        monitor='val/iou',       # Metric to monitor
        mode='max',              # Save model with highest IoU
        save_top_k=3,            # Save top 3 models
        save_last=True,          # Also save last checkpoint
        verbose=True,
    )
    
    # Learning rate monitor: Logs LR to TensorBoard
    lr_monitor = LearningRateMonitor(
        logging_interval='epoch',  # Log every epoch
        log_momentum=False,
    )
    
    # Early stopping: Stops training if validation loss doesn't improve
    early_stop_callback = EarlyStopping(
        monitor='val/loss',     # Metric to monitor
        patience=15,            # Number of epochs to wait
        mode='min',             # Stop if loss doesn't decrease
        verbose=True,
    )
    
    callbacks = [checkpoint_callback, lr_monitor, early_stop_callback]
    
    # ========================================================================
    # Step 4: Setup TensorBoard Logger
    # ========================================================================
    print("\nSetting up TensorBoard Logger...")
    print("=" * 80)
    print(f"TensorBoard logs will be saved to: {output_dir}/logs")
    print("To view logs, run: tensorboard --logdir={}/logs".format(output_dir))
    print("=" * 80)
    
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name='logs',
        default_hp_metric=False,
    )
    
    # ========================================================================
    # Step 5: Initialize PyTorch Lightning Trainer
    # ========================================================================
    print("\nInitializing PyTorch Lightning Trainer...")
    print("=" * 80)
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,        # Auto-detect GPU/CPU
        devices=devices,                # Number of GPUs/CPUs to use
        precision=precision,            # Mixed precision training
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,          # Log metrics every 10 steps
        check_val_every_n_epoch=1,     # Validate every epoch
        gradient_clip_val=1.0,         # Gradient clipping to prevent exploding gradients
        deterministic=False,           # For reproducibility (slower)
        benchmark=True,                # Enable cudnn benchmark for speed
    )
    
    # ========================================================================
    # Step 6: Start Training
    # ========================================================================
    print("\nStarting Training...")
    print("=" * 80)
    print(f"Max Epochs: {max_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Weight Decay: {weight_decay}")
    print(f"Image Size: {image_size}")
    print("=" * 80)
    
    # Fit the model
    trainer.fit(model, data_module)
    
    # ========================================================================
    # Step 7: Training Complete
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation IoU: {checkpoint_callback.best_model_score:.4f}")
    print("=" * 80)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """
    Main function to run the training pipeline.
    
    This is an example usage. In practice, you should:
    1. Prepare your dataset with images and masks
    2. Organize them into train/val splits
    3. Update the paths below to point to your data
    4. Adjust hyperparameters as needed
    """
    
    # Example configuration
    # Replace these paths with your actual data paths
    config = {
        # Data paths
        'train_images_dir': './data/train/images',
        'train_masks_dir': './data/train/masks',
        'val_images_dir': './data/val/images',
        'val_masks_dir': './data/val/masks',
        
        # Output directory
        'output_dir': './outputs/unet_training',
        
        # Model configuration
        'encoder_name': 'resnet34',  # Options: resnet34, resnet50, efficientnet-b0, etc.
        'encoder_weights': 'imagenet',  # Use ImageNet pre-trained weights
        'num_classes': 2,  # Number of segmentation classes (including background)
        
        # Training configuration
        'batch_size': 8,
        'num_workers': 4,
        'image_size': (256, 256),  # (height, width)
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'max_epochs': 100,
        
        # Hardware configuration
        'accelerator': 'auto',  # 'gpu', 'cpu', or 'auto'
        'devices': 1,  # Number of GPUs/CPUs
        'precision': '16-mixed',  # '32', '16-mixed', or 'bf16-mixed'
    }
    
    # Print configuration
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    for key, value in config.items():
        print(f"{key:20s}: {value}")
    print("=" * 80 + "\n")
    
    # Run training
    train(**config)


if __name__ == '__main__':
    main()
