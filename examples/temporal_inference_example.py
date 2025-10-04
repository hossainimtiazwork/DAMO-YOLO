#!/usr/bin/env python3
"""
Example: Using DAMO-YOLO with Temporal Features

This script demonstrates how to use the temporal backbone for inference
with sequences of frames.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


def load_temporal_model(config_path, checkpoint_path=None):
    """
    Load a temporal DAMO-YOLO model.
    
    Args:
        config_path: Path to temporal config file
        checkpoint_path: Optional path to checkpoint
        
    Returns:
        model: Loaded model
        cfg: Configuration object
    """
    from damo.config import parse_config
    from damo.base_models.core.damoyolo import DamoYOLO
    
    # Parse config
    cfg = parse_config(config_path)
    
    # Build model
    model = DamoYOLO(
        backbone=cfg.model.backbone,
        neck=cfg.model.neck,
        head=cfg.model.head
    )
    
    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    
    model.eval()
    return model, cfg


def preprocess_frame(frame, input_size=(640, 640)):
    """
    Preprocess a single frame.
    
    Args:
        frame: Input image (numpy array, RGB)
        input_size: Target size (H, W)
        
    Returns:
        Preprocessed tensor
    """
    # Resize (simple version without cv2)
    # In production, use cv2.resize for better quality
    if frame.shape[:2] != input_size:
        try:
            import cv2
            frame = cv2.resize(frame, input_size)
        except ImportError:
            # Fallback: skip resizing for demo
            if frame.shape[:2] == input_size:
                pass  # Already correct size
    
    # Convert to float and normalize
    frame = frame.astype(np.float32) / 255.0
    
    # HWC to CHW
    frame = np.transpose(frame, (2, 0, 1))
    
    return torch.from_numpy(frame)


def temporal_inference_from_frames(model, frames, device='cpu'):
    """
    Run inference on a sequence of frames.
    
    Args:
        model: Temporal DAMO-YOLO model
        frames: List of numpy arrays (RGB images)
        device: Device to run on
        
    Returns:
        Predictions from the model
    """
    # Preprocess all frames
    processed_frames = []
    for frame in frames:
        processed = preprocess_frame(frame)
        processed_frames.append(processed)
    
    # Stack into temporal tensor: (T, C, H, W)
    temporal_input = torch.stack(processed_frames, dim=0)
    
    # Add batch dimension: (1, T, C, H, W)
    temporal_input = temporal_input.unsqueeze(0)
    
    # Move to device
    temporal_input = temporal_input.to(device)
    model = model.to(device)
    
    # Inference
    with torch.no_grad():
        predictions = model(temporal_input)
    
    return predictions


def single_frame_inference(model, frame, device='cpu'):
    """
    Run inference on a single frame (for comparison).
    
    Args:
        model: DAMO-YOLO model (temporal or standard)
        frame: Numpy array (RGB image)
        device: Device to run on
        
    Returns:
        Predictions from the model
    """
    # Preprocess
    processed = preprocess_frame(frame)
    
    # Add batch dimension: (1, C, H, W)
    input_tensor = processed.unsqueeze(0)
    
    # Move to device
    input_tensor = input_tensor.to(device)
    model = model.to(device)
    
    # Inference
    with torch.no_grad():
        predictions = model(input_tensor)
    
    return predictions


def example_usage():
    """Example of using temporal features."""
    
    print("=" * 60)
    print("DAMO-YOLO Temporal Inference Example")
    print("=" * 60 + "\n")
    
    # Configuration
    config_path = 'configs/damoyolo_tinynasL45_L_temporal.py'
    num_frames = 3  # Should match config
    
    print(f"Configuration: {config_path}")
    print(f"Number of frames: {num_frames}\n")
    
    # Create dummy frames for demonstration
    # In real use, load from video or image sequence
    print("Creating dummy frames...")
    frames = []
    for i in range(num_frames):
        # Create random RGB image
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        frames.append(frame)
    print(f"Created {len(frames)} frames of shape {frames[0].shape}\n")
    
    # Note: Model loading would require actual checkpoint
    # This is just to demonstrate the API
    print("Model Loading:")
    print("  model, cfg = load_temporal_model(config_path, checkpoint_path)")
    print("  (Skipped - requires actual checkpoint)\n")
    
    # Demonstrate preprocessing
    print("Preprocessing frames...")
    processed_frames = [preprocess_frame(f) for f in frames]
    temporal_input = torch.stack(processed_frames, dim=0).unsqueeze(0)
    print(f"  Temporal input shape: {temporal_input.shape}")
    print(f"  Format: (batch, time, channels, height, width)\n")
    
    # Show inference call structure
    print("Inference (pseudocode):")
    print("  predictions = temporal_inference_from_frames(model, frames)")
    print("  or")
    print("  predictions = model(temporal_input)\n")
    
    print("=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nTo use with real model:")
    print("1. Train model: python tools/train.py -f configs/damoyolo_tinynasL45_L_temporal.py")
    print("2. Load checkpoint in this script")
    print("3. Provide real frame sequences")


def load_frames_from_video(video_path, num_frames=3, start_frame=0):
    """
    Load frames from a video file.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to load
        start_frame: Starting frame index
        
    Returns:
        List of RGB numpy arrays
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    return frames


def load_frames_from_images(image_paths):
    """
    Load frames from a list of image paths.
    
    Args:
        image_paths: List of paths to images
        
    Returns:
        List of RGB numpy arrays
    """
    import cv2
    
    frames = []
    for path in image_paths:
        frame = cv2.imread(path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    return frames


if __name__ == '__main__':
    example_usage()
