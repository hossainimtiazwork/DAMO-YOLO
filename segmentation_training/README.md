# Semantic Segmentation Training Pipeline

A comprehensive, production-ready training pipeline for semantic segmentation using:
- **SMP (Segmentation Models PyTorch)** - UNet architecture with various encoder backbones
- **PyTorch Lightning** - Modern training framework with best practices
- **Albumentations** - Advanced image augmentation library
- **Configurable Optimizer & Scheduler** - AdamW optimizer with Cosine Annealing scheduler
- **TensorBoard** - Rich visualization and logging

## 🎯 Features

### 1. **Modular Architecture**
- Clean separation of data loading, model definition, and training logic
- Easy to extend and customize for different tasks
- Well-documented code with extensive inline comments

### 2. **Advanced Data Augmentation**
Implemented using Albumentations library:
- **Geometric transformations**: Horizontal/vertical flips, rotations, shifts, scaling
- **Color transformations**: Brightness, contrast, hue, saturation adjustments
- **Noise and blur**: Gaussian noise, Gaussian blur
- **Proper normalization**: ImageNet statistics for pre-trained models

### 3. **Flexible Model Architecture**
Using SMP (Segmentation Models PyTorch):
- Multiple encoder options (ResNet, EfficientNet, DenseNet, MobileNet, VGG, etc.)
- Pre-trained weights from ImageNet
- UNet decoder architecture optimized for segmentation

### 4. **Training Best Practices**
- **Mixed precision training** for faster computation
- **Gradient clipping** to prevent exploding gradients
- **Early stopping** to prevent overfitting
- **Model checkpointing** to save best models
- **Learning rate monitoring** via TensorBoard
- **Cosine annealing scheduler** for smooth LR decay

### 5. **Comprehensive Logging**
TensorBoard integration for:
- Training and validation metrics (loss, IoU, accuracy)
- Learning rate tracking
- Sample predictions visualization
- Real-time monitoring during training

## 📋 Requirements

### Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

The training pipeline requires:
- `segmentation-models-pytorch` - UNet and encoder backbones
- `pytorch-lightning>=2.0.0` - Training framework
- `albumentations>=1.3.0` - Data augmentation
- `tensorboard` - Logging and visualization
- `torch`, `torchvision` - PyTorch
- `opencv-python`, `numpy`, `Pillow` - Image processing

## 📁 Data Preparation

### Directory Structure

Organize your dataset as follows:

```
data/
├── train/
│   ├── images/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   └── masks/
│       ├── image_001.png
│       ├── image_002.png
│       └── ...
└── val/
    ├── images/
    │   ├── image_101.jpg
    │   ├── image_102.jpg
    │   └── ...
    └── masks/
        ├── image_101.png
        ├── image_102.png
        └── ...
```

### Data Format

- **Images**: RGB images in common formats (JPG, PNG)
- **Masks**: Grayscale images where pixel values represent class labels
  - For binary segmentation: 0 (background), 1 (foreground)
  - For multi-class: 0, 1, 2, ..., N-1 (where N is the number of classes)
- **Naming**: Image and mask files should have the same name (extension can differ)

## 🚀 Usage

### Quick Start

1. **Prepare your dataset** following the structure above

2. **Update configuration** in `config.yaml`:
   ```yaml
   data:
     train_images_dir: "./data/train/images"
     train_masks_dir: "./data/train/masks"
     val_images_dir: "./data/val/images"
     val_masks_dir: "./data/val/masks"
   
   model:
     encoder_name: "resnet34"
     num_classes: 2  # Adjust based on your task
   ```

3. **Run training**:
   ```bash
   python train_unet.py
   ```

### Advanced Usage

#### Modify Training Parameters

Edit the `train()` function call in `train_unet.py`:

```python
config = {
    # Model
    'encoder_name': 'resnet50',  # Change encoder
    'num_classes': 5,  # Multi-class segmentation
    
    # Training
    'batch_size': 16,  # Larger batch size
    'learning_rate': 2e-4,  # Higher learning rate
    'max_epochs': 150,
    
    # Image size
    'image_size': (512, 512),  # Larger images
    
    # Hardware
    'devices': 2,  # Multi-GPU training
    'precision': '16-mixed',  # Mixed precision
}
```

#### Available Encoder Options

The pipeline supports multiple encoder backbones from SMP:

**ResNet family:**
- `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`

**EfficientNet family:**
- `efficientnet-b0` to `efficientnet-b7`

**DenseNet family:**
- `densenet121`, `densenet169`, `densenet201`

**Other architectures:**
- `mobilenet_v2`
- `vgg16`, `vgg19`
- And many more...

See [SMP documentation](https://github.com/qubvel/segmentation_models.pytorch) for full list.

## 📊 Monitoring Training

### TensorBoard

Launch TensorBoard to monitor training progress:

```bash
tensorboard --logdir=./outputs/unet_training/logs
```

Then open your browser and navigate to `http://localhost:6006`

### What You Can Monitor

1. **Scalars**:
   - Training/validation loss
   - Training/validation IoU (Intersection over Union)
   - Training/validation pixel accuracy
   - Learning rate over time

2. **Images**:
   - Sample input images
   - Ground truth masks
   - Predicted masks

3. **Graphs**:
   - Model architecture visualization

## 🎛️ Configuration

### Key Hyperparameters

#### Learning Rate
```python
learning_rate: float = 1e-4  # Default: 0.0001
```
- Start with `1e-4` for most cases
- Increase to `2e-4` or `5e-4` for faster convergence
- Decrease to `5e-5` or `1e-5` for fine-tuning

#### Batch Size
```python
batch_size: int = 8  # Default: 8
```
- Larger batch sizes (16, 32) for better gradient estimates
- Reduce if you encounter out-of-memory errors
- Consider gradient accumulation for effective larger batch sizes

#### Weight Decay
```python
weight_decay: float = 1e-4  # Default: 0.0001
```
- Regularization to prevent overfitting
- Typical range: `1e-5` to `1e-4`

#### Image Size
```python
image_size: Tuple[int, int] = (256, 256)  # (height, width)
```
- Larger images capture more detail but require more memory
- Common sizes: `(256, 256)`, `(384, 384)`, `(512, 512)`

### Optimizer Configuration

Current implementation uses **AdamW**:
```python
optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
    betas=(0.9, 0.999),
    eps=1e-8,
)
```

**Why AdamW?**
- Decoupled weight decay for better regularization
- Works well with both small and large learning rates
- Generally more stable than standard Adam

### Scheduler Configuration

Current implementation uses **CosineAnnealingLR**:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max_epochs,
    eta_min=1e-6,
)
```

**Benefits:**
- Smooth learning rate decay
- Gradual reduction helps fine-tune model
- Follows cosine curve from initial LR to minimum LR

## 🔧 Customization

### Custom Dataset

To use your own dataset, modify the `SegmentationDataset` class in `train_unet.py`:

```python
class CustomDataset(SegmentationDataset):
    def __getitem__(self, idx):
        # Your custom loading logic
        image = self.load_custom_image(idx)
        mask = self.load_custom_mask(idx)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask
```

### Custom Augmentations

Modify `get_train_transform()` in `SegmentationDataModule`:

```python
def get_train_transform(self):
    return A.Compose([
        A.Resize(height=self.image_size[0], width=self.image_size[1]),
        
        # Add your custom augmentations
        A.RandomCrop(width=224, height=224),
        A.ElasticTransform(p=0.5),
        A.GridDistortion(p=0.5),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
```

### Custom Loss Function

Modify the loss function in `SegmentationModel`:

```python
# Example: Dice Loss
import segmentation_models_pytorch as smp

self.criterion = smp.losses.DiceLoss(mode='multiclass')

# Example: Combined Loss
self.criterion = smp.losses.DiceLoss() + nn.CrossEntropyLoss()
```

### Custom Metrics

Add custom metrics in `compute_metrics()`:

```python
def compute_metrics(self, preds, targets):
    # Your custom metric calculation
    dice_score = self.compute_dice(preds, targets)
    f1_score = self.compute_f1(preds, targets)
    
    return {
        'dice': dice_score,
        'f1': f1_score,
    }
```

## 🎓 Training Tips

### 1. Start Simple
- Begin with a small model (`resnet18`, `efficientnet-b0`)
- Use lower resolution (`256x256`)
- Train for fewer epochs to validate pipeline

### 2. Use Pre-trained Weights
- Always use `encoder_weights='imagenet'` unless you have a good reason not to
- Pre-trained weights significantly improve convergence

### 3. Monitor Validation Metrics
- Watch for overfitting (train loss decreasing, val loss increasing)
- Use early stopping to prevent wasting compute
- Check IoU and pixel accuracy, not just loss

### 4. Experiment with Augmentations
- Start with basic augmentations (flips, rotations)
- Gradually add more complex ones
- Too much augmentation can hurt performance

### 5. Learning Rate Scheduling
- Cosine annealing works well for most cases
- Consider warmup for very deep models
- Monitor LR in TensorBoard to ensure proper decay

### 6. Mixed Precision Training
- Use `precision='16-mixed'` on modern GPUs (Volta, Turing, Ampere)
- Can speed up training by 2-3x with minimal accuracy loss
- Reduces memory usage, allowing larger batch sizes

## 📈 Evaluation Metrics

### Intersection over Union (IoU)
- Primary metric for segmentation quality
- Range: 0 (no overlap) to 1 (perfect overlap)
- Computed per-class and averaged

### Pixel Accuracy
- Percentage of correctly classified pixels
- Simple but can be misleading with class imbalance
- Use IoU as primary metric

### Dice Coefficient
- Similar to IoU, alternative similarity measure
- Harmonic mean of precision and recall
- Range: 0 to 1

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors
1. Reduce `batch_size`
2. Reduce `image_size`
3. Use smaller encoder (e.g., `resnet18` instead of `resnet50`)
4. Enable gradient accumulation
5. Use mixed precision training

### Poor Convergence
1. Check data quality and labels
2. Increase learning rate
3. Use more augmentations
4. Train for more epochs
5. Try different encoder backbone

### Overfitting
1. Increase weight decay
2. Add more augmentations
3. Use dropout in decoder
4. Get more training data
5. Reduce model size

### Training Too Slow
1. Enable mixed precision (`precision='16-mixed'`)
2. Increase `num_workers` for data loading
3. Use smaller images
4. Use `benchmark=True` in Trainer
5. Consider multi-GPU training

## 📚 Additional Resources

### Documentation
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [Albumentations](https://albumentations.ai/docs/)

### Tutorials
- [UNet Architecture](https://arxiv.org/abs/1505.04597)
- [Data Augmentation for Segmentation](https://albumentations.ai/docs/examples/example_kaggle_salt/)
- [PyTorch Lightning Tutorial](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)

## 📝 Example Output

After training completes, you'll find:

```
outputs/unet_training/
├── logs/                          # TensorBoard logs
│   └── version_0/
│       ├── events.out.tfevents...
│       └── hparams.yaml
└── checkpoints/                   # Model checkpoints
    ├── unet-epoch=10-val_iou=0.8523.ckpt
    ├── unet-epoch=25-val_iou=0.8891.ckpt
    ├── unet-epoch=42-val_iou=0.9124.ckpt
    └── last.ckpt
```

## 🤝 Contributing

Feel free to extend this pipeline with:
- Additional architectures (DeepLabV3, FPN, etc.)
- More loss functions (Focal Loss, Tversky Loss, etc.)
- Advanced schedulers (OneCycleLR, ReduceLROnPlateau)
- Multi-GPU/distributed training examples

## 📄 License

This training pipeline is part of the DAMO-YOLO project.

## 🙏 Acknowledgments

- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch) for model architectures
- [PyTorch Lightning](https://lightning.ai/) for training framework
- [Albumentations](https://albumentations.ai/) for augmentation library
