#!/usr/bin/env python3
"""
Example usage script demonstrating the segmentation training pipeline.

This script shows how to:
1. Set up a simple synthetic dataset
2. Configure the training pipeline
3. Run training with proper settings
4. Load and use a trained model for inference

Note: This is a minimal example. For real applications, replace the
synthetic data with your actual dataset.
"""

import os
import numpy as np
import cv2
from train_unet import train


def create_sample_dataset(base_dir: str = './sample_data', num_samples: int = 50):
    """
    Create a simple synthetic dataset for demonstration.
    
    This function generates random images and corresponding binary masks
    to demonstrate the pipeline. Replace this with your actual data loading.
    
    Args:
        base_dir (str): Base directory to save sample data
        num_samples (int): Number of samples to generate
    """
    print("Creating sample synthetic dataset...")
    print("=" * 80)
    
    # Create directory structure
    dirs = [
        os.path.join(base_dir, 'train', 'images'),
        os.path.join(base_dir, 'train', 'masks'),
        os.path.join(base_dir, 'val', 'images'),
        os.path.join(base_dir, 'val', 'masks'),
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Generate training samples
    print(f"Generating {num_samples} training samples...")
    for i in range(num_samples):
        # Create synthetic RGB image (256x256)
        # In practice, load your real images here
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        # Create synthetic binary segmentation mask
        # In practice, load your real masks here
        # This creates a circular mask in the center
        y, x = np.ogrid[:256, :256]
        center_y, center_x = 128 + np.random.randint(-40, 40), 128 + np.random.randint(-40, 40)
        radius = np.random.randint(40, 80)
        mask = ((x - center_x)**2 + (y - center_y)**2 <= radius**2).astype(np.uint8)
        
        # Save image and mask
        image_path = os.path.join(base_dir, 'train', 'images', f'sample_{i:04d}.jpg')
        mask_path = os.path.join(base_dir, 'train', 'masks', f'sample_{i:04d}.png')
        
        cv2.imwrite(image_path, image)
        cv2.imwrite(mask_path, mask)
    
    # Generate validation samples
    num_val_samples = num_samples // 5  # 20% for validation
    print(f"Generating {num_val_samples} validation samples...")
    for i in range(num_val_samples):
        # Create synthetic image and mask
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        y, x = np.ogrid[:256, :256]
        center_y, center_x = 128 + np.random.randint(-40, 40), 128 + np.random.randint(-40, 40)
        radius = np.random.randint(40, 80)
        mask = ((x - center_x)**2 + (y - center_y)**2 <= radius**2).astype(np.uint8)
        
        # Save image and mask
        image_path = os.path.join(base_dir, 'val', 'images', f'sample_{i:04d}.jpg')
        mask_path = os.path.join(base_dir, 'val', 'masks', f'sample_{i:04d}.png')
        
        cv2.imwrite(image_path, image)
        cv2.imwrite(mask_path, mask)
    
    print(f"\nSample dataset created at: {base_dir}")
    print(f"  - Training samples: {num_samples}")
    print(f"  - Validation samples: {num_val_samples}")
    print("=" * 80 + "\n")


def run_training_example():
    """
    Run a complete training example with synthetic data.
    
    This demonstrates the full pipeline:
    1. Dataset creation
    2. Configuration
    3. Training
    4. Model checkpointing
    """
    
    print("\n" + "=" * 80)
    print("SEGMENTATION TRAINING PIPELINE - EXAMPLE USAGE")
    print("=" * 80 + "\n")
    
    # Step 1: Create sample dataset
    # In practice, you would skip this and use your own data
    sample_data_dir = './sample_data'
    if not os.path.exists(sample_data_dir):
        create_sample_dataset(base_dir=sample_data_dir, num_samples=50)
    else:
        print(f"Using existing sample dataset at: {sample_data_dir}\n")
    
    # Step 2: Configure training parameters
    config = {
        # ====================================================================
        # Data Configuration
        # ====================================================================
        'train_images_dir': os.path.join(sample_data_dir, 'train', 'images'),
        'train_masks_dir': os.path.join(sample_data_dir, 'train', 'masks'),
        'val_images_dir': os.path.join(sample_data_dir, 'val', 'images'),
        'val_masks_dir': os.path.join(sample_data_dir, 'val', 'masks'),
        
        # Output directory for logs and checkpoints
        'output_dir': './outputs/example_training',
        
        # ====================================================================
        # Model Configuration
        # ====================================================================
        # Encoder backbone: Use lightweight model for quick example
        # Options: resnet18, resnet34, resnet50, efficientnet-b0, etc.
        'encoder_name': 'resnet18',
        
        # Pre-trained weights: Use ImageNet weights for transfer learning
        'encoder_weights': 'imagenet',
        
        # Number of classes: Binary segmentation (background + foreground)
        'num_classes': 2,
        
        # ====================================================================
        # Training Configuration
        # ====================================================================
        # Batch size: Number of samples per batch
        # Reduce if you get out-of-memory errors
        'batch_size': 4,
        
        # Number of data loading workers
        # Set to 0 for single-process loading (use on Windows if multiprocessing issues)
        'num_workers': 2,
        
        # Input image size (height, width)
        # Smaller size = faster training, less memory usage
        'image_size': (256, 256),
        
        # Initial learning rate
        # Start with 1e-4, adjust based on convergence
        'learning_rate': 1e-4,
        
        # Weight decay for L2 regularization
        # Helps prevent overfitting
        'weight_decay': 1e-4,
        
        # Maximum number of training epochs
        # For demo: use 10-20 epochs
        # For real training: 50-100+ epochs
        'max_epochs': 10,
        
        # ====================================================================
        # Hardware Configuration
        # ====================================================================
        # Accelerator: 'auto' detects GPU/CPU automatically
        'accelerator': 'auto',
        
        # Number of devices: 1 for single GPU/CPU
        'devices': 1,
        
        # Precision: '16-mixed' for mixed precision (faster on modern GPUs)
        # Use '32' for full precision if issues occur
        'precision': '32',  # Use '16-mixed' for faster training on GPU
    }
    
    # Step 3: Print configuration
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print("\nData:")
    print(f"  Train images: {config['train_images_dir']}")
    print(f"  Train masks:  {config['train_masks_dir']}")
    print(f"  Val images:   {config['val_images_dir']}")
    print(f"  Val masks:    {config['val_masks_dir']}")
    print(f"\nModel:")
    print(f"  Encoder:      {config['encoder_name']}")
    print(f"  Weights:      {config['encoder_weights']}")
    print(f"  Classes:      {config['num_classes']}")
    print(f"\nTraining:")
    print(f"  Batch size:   {config['batch_size']}")
    print(f"  Epochs:       {config['max_epochs']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Image size:   {config['image_size']}")
    print(f"\nHardware:")
    print(f"  Accelerator:  {config['accelerator']}")
    print(f"  Devices:      {config['devices']}")
    print(f"  Precision:    {config['precision']}")
    print("=" * 80 + "\n")
    
    # Step 4: Run training
    print("Starting training pipeline...")
    print("=" * 80 + "\n")
    
    try:
        train(**config)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        raise
    
    # Step 5: Training complete
    print("\n" + "=" * 80)
    print("EXAMPLE TRAINING COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. View training logs:")
    print(f"   tensorboard --logdir={config['output_dir']}/logs")
    print("\n2. Use trained model for inference:")
    print("   python inference.py \\")
    print(f"       --checkpoint {config['output_dir']}/checkpoints/last.ckpt \\")
    print("       --image path/to/your/image.jpg \\")
    print("       --visualize")
    print("\n3. Load checkpoint in Python:")
    print("   from train_unet import SegmentationModel")
    print(f"   model = SegmentationModel.load_from_checkpoint('{config['output_dir']}/checkpoints/last.ckpt')")
    print("=" * 80 + "\n")


def quick_inference_example():
    """
    Example of how to use a trained model for inference.
    
    This demonstrates loading a checkpoint and making predictions.
    """
    print("\n" + "=" * 80)
    print("INFERENCE EXAMPLE")
    print("=" * 80 + "\n")
    
    # Note: This assumes you have already trained a model
    checkpoint_path = './outputs/example_training/checkpoints/last.ckpt'
    
    if not os.path.exists(checkpoint_path):
        print("No trained checkpoint found.")
        print("Please run training first using run_training_example()")
        return
    
    print("Loading trained model...")
    
    # Import required modules
    from train_unet import SegmentationModel
    import torch
    
    # Load model
    model = SegmentationModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"Model loaded on {device}")
    print(f"Model architecture: UNet with {model.hparams.encoder_name} encoder")
    print(f"Number of classes: {model.hparams.num_classes}")
    
    print("\nTo run inference on an image, use:")
    print("python inference.py \\")
    print(f"    --checkpoint {checkpoint_path} \\")
    print("    --image path/to/your/image.jpg \\")
    print("    --visualize")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    """
    Main entry point for the example script.
    
    Uncomment the function you want to run:
    - run_training_example(): Run full training pipeline with synthetic data
    - quick_inference_example(): Show how to use trained model
    """
    
    # Run the training example
    # This will create synthetic data and train a model for demonstration
    run_training_example()
    
    # Optionally, run the inference example after training
    # Uncomment the line below to see how to use the trained model
    # quick_inference_example()
