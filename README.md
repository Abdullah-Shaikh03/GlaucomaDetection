# Glaucoma Detection using Deep Learning

## Project Overview
This project implements a deep learning solution for automated detection of glaucoma from retinal images. It combines a custom HybridNet architecture with attention mechanisms and advanced augmentation techniques to handle class imbalance in medical imaging data.

## Dataset Availability

1. [https://drive.google.com/drive/folders/1hfjjTqPAWyVOIhMe-t37NL2RKV_5J7Zz?usp=sharing] ``` Open Access ```
<!-- 2. [https://drive.google.com/drive/folders/1P7ClDUUHXE40cinDujZZApUdLBugltEa?usp=sharing] ``` Restricted access ``` -->


## Key Features

### Model Architecture
- Custom HybridNet implementation combining VGG, DenseNet, ResNet, and Inception architectures
- Squeeze-and-Excitation blocks for channel attention
- Spatial attention mechanisms for feature refinement
- Stochastic depth for improved regularization
- Batch normalization and dropout layers

### Data Processing
- Advanced augmentation pipeline with:
  - Class-specific transformations
  - Mixup and CutMix augmentation
  - Geometric transformations (rotation, scaling, flips)
  - Color space augmentations
  - Adaptive histogram equalization
  - Gaussian noise injection
- Balanced sampling strategy for handling class imbalance

### Training
- Focal Loss implementation for handling class imbalance
- Learning rate scheduling with OneCycleLR
- Early stopping and model checkpointing
- TensorBoard integration for monitoring:
  - Training/validation metrics
  - Model gradients
  - Feature visualizations
  - Confusion matrices

## Installation

1. Clone the repository:
```bash
git clone https://gitlab.com/rp20241/GlaucomaDetection.git
cd glaucoma-detection
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# On Windows:
venvScriptsactivate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation
1. Place your raw retinal images in `data/raw/`
2. Run preprocessing:
```bash
python Scripts/Analysis.py --preprocess
```

### Training
1. Configure training parameters in `Scripts/Train.py`
2. Start training:
```bash
python Scripts/Train.py
```

### Visualization
Generate visualizations of model predictions:
```bash
python Scripts/visualize.py --model-path models/best_model.pth
```

## Model Configuration

### Network Architecture
- Input Resolution: 224x224x3
- Backbone: Custom HybridNet (VGG, DenseNet, ResNet, Inception hybrid)
- Attention Mechanisms:
  - Channel attention (SE blocks)
  - Spatial attention
  - Feature refinement modules

### Training Parameters
- Batch Size: 32
- Initial Learning Rate: 1e-3
- Weight Decay: 1e-4
- Training Epochs: 100
- Early Stopping Patience: 15
- Loss Function: Focal Loss
- Optimizer: AdamW
- Scheduler: OneCycleLR

## Scripts Description

- `Analysis.py`: Data analysis and preprocessing utilities
- `Augmenter.py`: Implements data augmentation pipeline
- `DataLoader.py`: Custom dataset implementations and data loading
- `losses.py`: Custom loss functions including Focal Loss
- `Model.py`: Neural network architecture implementation
- `Train.py`: Main training loop and configuration
- `visualization.py`: Training visualization tools
- `visualize.py`: Model prediction visualization

## Development Guide

### Setting Up Development Environment
1. Fork the repository
2. Create a new branch for your feature
3. Install development dependencies:
```bash
pip install -r requirements.txt
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Include docstrings for all functions/classes
- Add comments for complex logic

## Performance Metrics

### Resource Requirements
- Minimum RAM: 16GB
- Recommended GPU: 8GB VRAM
- Storage: 50GB for dataset and models

### Training Time
- Single GPU (V100): ~8 hours
- Multi-GPU scaling available

### Model Size
- Parameters: ~45.5M (may vary slightly due to hybrid architecture)
- Disk Space: ~175MB

## Troubleshooting

Common issues and solutions:

1. Out of Memory (OOM):
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

2. Slow Training:
   - Check data loading pipeline
   - Enable GPU acceleration
   - Optimize augmentation pipeline

3. Poor Convergence:
   - Verify data preprocessing
   - Check learning rate schedule
   - Inspect loss curves

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

Contribution areas:
- Model architecture improvements
- Data augmentation techniques
- Performance optimizations
- Documentation
- Test coverage
- Bug fixes

## Future Improvements

Planned enhancements:
- [ ] Multi-GPU training support
- [ ] Model quantization
- [ ] ONNX export
- [ ] Additional data augmentation techniques
- [ ] Model interpretability tools

## Support

For issues and questions:
1. Check existing issues
2. Create a new issue with:
   - Environment details
   - Steps to reproduce
   - Expected vs actual behavior
   - Relevant logs

## Acknowledgments

This project builds upon various open-source libraries and research papers. Special thanks to:
- PyTorch team
- torchvision contributors
- Medical imaging research community

<!-- export HSA_OVERRIDE_GFX_VERSION=10.3.0 -->

