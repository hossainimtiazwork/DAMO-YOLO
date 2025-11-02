# Quick Start Guide - Segmentation Training Pipeline

## 🚀 Getting Started in 5 Minutes

This guide will help you get the segmentation training pipeline up and running quickly.

## Step 1: Install Dependencies

```bash
# Navigate to the repository
cd DAMO-YOLO

# Install required packages
pip install segmentation-models-pytorch pytorch-lightning>=2.0.0 albumentations>=1.3.0 tensorboard

# Or install from requirements.txt
pip install -r requirements.txt
```

## Step 2: Prepare Your Data

Organize your dataset in the following structure:

```
your_data/
├── train/
│   ├── images/    # Training images (.jpg, .png)
│   └── masks/     # Training masks (.png)
└── val/
    ├── images/    # Validation images
    └── masks/     # Validation masks
```

**Important:** 
- Masks should be grayscale images where pixel values represent class labels (0, 1, 2, ...)
- Image and mask filenames should match (extensions can differ)

## Step 3: Try the Example

Run the example with synthetic data to verify installation:

```bash
cd segmentation_training
python example_usage.py
```

This will:
- Generate synthetic training data
- Train a small UNet model for 10 epochs
- Save checkpoints and logs

## Step 4: Train on Your Data

Edit `train_unet.py` main function or create your own script:

```python
from train_unet import train

# Configure training
config = {
    'train_images_dir': './your_data/train/images',
    'train_masks_dir': './your_data/train/masks',
    'val_images_dir': './your_data/val/images',
    'val_masks_dir': './your_data/val/masks',
    'output_dir': './outputs/my_training',
    'encoder_name': 'resnet34',
    'num_classes': 2,  # Adjust for your task
    'batch_size': 8,
    'max_epochs': 100,
}

# Run training
train(**config)
```

## Step 5: Monitor Training

Open TensorBoard to view training progress:

```bash
tensorboard --logdir=./outputs/my_training/logs
```

Then open your browser to `http://localhost:6006`

## Step 6: Run Inference

Use the trained model for predictions:

```bash
python inference.py \
    --checkpoint ./outputs/my_training/checkpoints/best.ckpt \
    --image path/to/test_image.jpg \
    --visualize \
    --output-dir ./predictions
```

## 📊 What You'll See

### During Training
- **Console output**: Epoch progress, loss, IoU, accuracy
- **TensorBoard**: Real-time metrics, learning rate curves, sample predictions

### After Training
- **Checkpoints**: Best models saved in `outputs/*/checkpoints/`
- **Logs**: TensorBoard logs in `outputs/*/logs/`
- **Predictions**: Segmentation masks in your output directory

## ⚙️ Common Configurations

### Binary Segmentation (e.g., Road Detection)
```python
config = {
    'num_classes': 2,           # Background + road
    'encoder_name': 'resnet34',
    'batch_size': 8,
    'learning_rate': 1e-4,
}
```

### Multi-class Segmentation (e.g., Cityscapes)
```python
config = {
    'num_classes': 19,          # Multiple classes
    'encoder_name': 'resnet50',
    'batch_size': 4,            # Smaller batch due to more classes
    'learning_rate': 2e-4,
}
```

### High-Resolution Images
```python
config = {
    'image_size': (512, 512),   # Larger images
    'batch_size': 4,            # Reduce batch size
    'encoder_name': 'resnet34', # Or lighter encoder
    'precision': '16-mixed',    # Use mixed precision
}
```

### Fast Training (GPU with Mixed Precision)
```python
config = {
    'batch_size': 16,
    'precision': '16-mixed',    # 2-3x speedup on modern GPUs
    'num_workers': 8,
    'accelerator': 'gpu',
}
```

## 🔧 Troubleshooting

### Out of Memory
```python
# Reduce any of these:
'batch_size': 4,           # Smaller batches
'image_size': (256, 256),  # Smaller images
'encoder_name': 'resnet18' # Lighter model
```

### Slow Training
```python
# Try these optimizations:
'precision': '16-mixed',   # Mixed precision (GPU only)
'num_workers': 4,          # More data loading workers
'batch_size': 16,          # Larger batches if memory allows
```

### Poor Accuracy
- Check your data quality and labels
- Try pre-trained weights: `encoder_weights='imagenet'`
- Increase training epochs
- Add more data augmentation
- Try a deeper encoder (resnet50, resnet101)

## 📚 Next Steps

1. **Read the full documentation**: See `README.md` for comprehensive guide
2. **Customize augmentations**: Modify `get_train_transform()` in `train_unet.py`
3. **Try different architectures**: Change `encoder_name` to explore options
4. **Experiment with hyperparameters**: Adjust learning rate, batch size, etc.
5. **Export for deployment**: Convert trained model to ONNX or TorchScript

## 💡 Key Tips

1. **Always start with pre-trained weights** (`encoder_weights='imagenet'`)
2. **Monitor both loss and IoU** - IoU is more meaningful for segmentation
3. **Use TensorBoard** to visualize sample predictions during training
4. **Save time with early stopping** - enabled by default
5. **Mixed precision is your friend** - 2-3x speedup on modern GPUs

## 🆘 Getting Help

- Check the detailed `README.md` for more information
- Review inline comments in `train_unet.py` for implementation details
- See `example_usage.py` for a working example
- Refer to `config.yaml` for all configurable parameters

Happy training! 🎉
