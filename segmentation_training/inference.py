#!/usr/bin/env python3
"""
Inference script for trained UNet segmentation model.

This script demonstrates how to:
1. Load a trained checkpoint
2. Process input images
3. Generate segmentation predictions
4. Visualize and save results
"""

import os
import argparse
from typing import Tuple, Optional

import torch
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from train_unet import SegmentationModel


def load_model(checkpoint_path: str, device: str = 'cuda') -> SegmentationModel:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        device (str): Device to load model on ('cuda' or 'cpu')
    
    Returns:
        SegmentationModel: Loaded model in evaluation mode
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Load model from checkpoint
    # PyTorch Lightning automatically handles hyperparameters
    model = SegmentationModel.load_from_checkpoint(checkpoint_path)
    
    # Move model to specified device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model


def get_preprocessing_transform(image_size: Tuple[int, int] = (256, 256)) -> A.Compose:
    """
    Get preprocessing transform for inference.
    
    Should match the validation transform used during training
    (resize and normalize only, no augmentations).
    
    Args:
        image_size (tuple): Target image size (height, width)
    
    Returns:
        albumentations.Compose: Preprocessing pipeline
    """
    return A.Compose([
        # Resize to model input size
        A.Resize(height=image_size[0], width=image_size[1]),
        
        # Normalize using ImageNet statistics (must match training)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        
        # Convert to PyTorch tensor
        ToTensorV2(),
    ])


def load_and_preprocess_image(
    image_path: str,
    transform: A.Compose,
) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
    """
    Load and preprocess an image for inference.
    
    Args:
        image_path (str): Path to input image
        transform (A.Compose): Preprocessing transform
    
    Returns:
        tuple: (preprocessed_tensor, original_image, original_size)
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Store original image and size for visualization
    original_image = image.copy()
    original_size = (image.shape[0], image.shape[1])  # (height, width)
    
    # Apply preprocessing
    transformed = transform(image=image)
    preprocessed_tensor = transformed['image']
    
    # Add batch dimension: [C, H, W] -> [1, C, H, W]
    preprocessed_tensor = preprocessed_tensor.unsqueeze(0)
    
    return preprocessed_tensor, original_image, original_size


@torch.no_grad()  # Disable gradient computation for inference
def predict(
    model: SegmentationModel,
    image_tensor: torch.Tensor,
    device: str = 'cuda',
) -> np.ndarray:
    """
    Generate segmentation prediction for an image.
    
    Args:
        model (SegmentationModel): Trained model
        image_tensor (torch.Tensor): Preprocessed image tensor [1, C, H, W]
        device (str): Device to run inference on
    
    Returns:
        np.ndarray: Predicted mask [H, W] with class indices
    """
    # Move input to device
    image_tensor = image_tensor.to(device)
    
    # Forward pass
    logits = model(image_tensor)  # [1, num_classes, H, W]
    
    # Convert logits to class predictions
    # Take argmax over class dimension
    pred_mask = torch.argmax(logits, dim=1)  # [1, H, W]
    
    # Remove batch dimension and convert to numpy
    pred_mask = pred_mask.squeeze(0).cpu().numpy()  # [H, W]
    
    return pred_mask


def resize_mask(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize mask to target size using nearest neighbor interpolation.
    
    Args:
        mask (np.ndarray): Input mask [H, W]
        target_size (tuple): Target size (height, width)
    
    Returns:
        np.ndarray: Resized mask
    """
    # Use INTER_NEAREST to preserve class labels
    resized_mask = cv2.resize(
        mask.astype(np.uint8),
        (target_size[1], target_size[0]),  # cv2 uses (width, height)
        interpolation=cv2.INTER_NEAREST
    )
    return resized_mask


def visualize_prediction(
    image: np.ndarray,
    pred_mask: np.ndarray,
    num_classes: int,
    save_path: Optional[str] = None,
    gt_mask: Optional[np.ndarray] = None,
):
    """
    Visualize segmentation prediction.
    
    Creates a figure with:
    - Original image
    - Ground truth mask (if provided)
    - Predicted mask
    - Overlay of prediction on image
    
    Args:
        image (np.ndarray): Original RGB image [H, W, 3]
        pred_mask (np.ndarray): Predicted segmentation mask [H, W]
        num_classes (int): Number of classes
        save_path (str, optional): Path to save visualization
        gt_mask (np.ndarray, optional): Ground truth mask [H, W]
    """
    # Create color map for visualization
    # Each class gets a distinct color
    colors = plt.cm.get_cmap('tab10', num_classes)
    
    # Convert masks to colored images
    pred_colored = colors(pred_mask)[:, :, :3]  # Remove alpha channel
    
    # Create overlay: blend image with prediction
    alpha = 0.5  # Transparency factor
    overlay = (image / 255.0) * (1 - alpha) + pred_colored * alpha
    
    # Setup figure
    if gt_mask is not None:
        # Show ground truth if available
        gt_colored = colors(gt_mask)[:, :, :3]
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.ravel()
        
        # Plot ground truth
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(gt_colored)
        axes[1].set_title('Ground Truth Mask', fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(pred_colored)
        axes[2].set_title('Predicted Mask', fontsize=14)
        axes[2].axis('off')
        
        axes[3].imshow(overlay)
        axes[3].set_title('Overlay (Image + Prediction)', fontsize=14)
        axes[3].axis('off')
    else:
        # Show only prediction
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(pred_colored)
        axes[1].set_title('Predicted Mask', fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (Image + Prediction)', fontsize=14)
        axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_mask(mask: np.ndarray, save_path: str):
    """
    Save segmentation mask as grayscale image.
    
    Args:
        mask (np.ndarray): Segmentation mask [H, W]
        save_path (str): Path to save mask
    """
    # Check if mask values exceed uint8 range
    max_val = mask.max()
    if max_val > 255:
        print(f"Warning: Mask contains values up to {max_val}, which exceed uint8 range (0-255).")
        print("Consider using a 16-bit format if you have more than 256 classes.")
    
    # Convert to uint8 and save
    mask_uint8 = mask.astype(np.uint8)
    cv2.imwrite(save_path, mask_uint8)
    print(f"Mask saved to: {save_path}")


def main():
    """
    Main function for running inference.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Inference script for UNet segmentation')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.ckpt file)'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/predictions',
        help='Directory to save predictions'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        nargs=2,
        default=[256, 256],
        help='Model input size (height width)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create and save visualization'
    )
    parser.add_argument(
        '--gt-mask',
        type=str,
        default=None,
        help='Path to ground truth mask (optional, for visualization)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    print("=" * 80)
    print("INFERENCE CONFIGURATION")
    print("=" * 80)
    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Input image:   {args.image}")
    print(f"Output dir:    {args.output_dir}")
    print(f"Image size:    {args.image_size}")
    print(f"Device:        {args.device}")
    print("=" * 80 + "\n")
    
    # Load model
    model = load_model(args.checkpoint, device=args.device)
    num_classes = model.hparams.num_classes
    
    # Get preprocessing transform
    transform = get_preprocessing_transform(image_size=tuple(args.image_size))
    
    # Load and preprocess image
    print(f"Loading image: {args.image}")
    image_tensor, original_image, original_size = load_and_preprocess_image(
        args.image, transform
    )
    print(f"Original image size: {original_size}")
    
    # Run inference
    print("Running inference...")
    pred_mask = predict(model, image_tensor, device=args.device)
    print(f"Prediction shape: {pred_mask.shape}")
    
    # Resize prediction to original image size
    print(f"Resizing prediction to original size: {original_size}")
    pred_mask_resized = resize_mask(pred_mask, original_size)
    
    # Save prediction mask
    image_name = os.path.splitext(os.path.basename(args.image))[0]
    mask_save_path = os.path.join(args.output_dir, f"{image_name}_pred.png")
    save_mask(pred_mask_resized, mask_save_path)
    
    # Visualize if requested
    if args.visualize:
        print("Creating visualization...")
        
        # Load ground truth mask if provided
        gt_mask = None
        if args.gt_mask:
            gt_mask = cv2.imread(args.gt_mask, cv2.IMREAD_GRAYSCALE)
        
        # Create visualization
        vis_save_path = os.path.join(args.output_dir, f"{image_name}_vis.png")
        visualize_prediction(
            original_image,
            pred_mask_resized,
            num_classes,
            save_path=vis_save_path,
            gt_mask=gt_mask,
        )
    
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
    print(f"Prediction saved to: {mask_save_path}")
    if args.visualize:
        print(f"Visualization saved to: {vis_save_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
