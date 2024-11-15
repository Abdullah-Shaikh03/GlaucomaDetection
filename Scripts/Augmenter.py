import torch
import random
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from torchvision import transforms
from typing import Tuple, Optional, List
import logging

class CustomGaussianNoise:
    """Add Gaussian noise to tensor"""
    def __init__(self, mean: float = 0., std: float = 0.1):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class AdaptiveHistogramEqualization:
    """Apply adaptive histogram equalization"""
    def __call__(self, img: Image.Image) -> Image.Image:
        return ImageOps.equalize(img)

class RandomGammaCorrection:
    """Apply random gamma correction"""
    def __init__(self, gamma_range: Tuple[float, float] = (0.7, 1.3)):
        self.gamma_range = gamma_range
    
    def __call__(self, img: Image.Image) -> Image.Image:
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        return transforms.functional.adjust_gamma(img, gamma)

class RandomSharpness:
    """Apply random sharpness"""
    def __init__(self, sharpness_range: Tuple[float, float] = (0.8, 1.5)):
        self.sharpness_range = sharpness_range
    
    def __call__(self, img: Image.Image) -> Image.Image:
        factor = random.uniform(self.sharpness_range[0], self.sharpness_range[1])
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)

def get_enhanced_augmentation_transform(
    is_minority_class: bool = False,
    input_size: int = 256,
    additional_transforms: Optional[List] = None
) -> transforms.Compose:
    """
    Get enhanced augmentation transforms with special handling for minority class
    
    Args:
        is_minority_class: Whether these transforms are for the minority class
        input_size: Input image size
        additional_transforms: Additional transform operations to include
    
    Returns:
        Composed transformation pipeline
    """
    # Base transforms for all classes
    base_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
        transforms.RandomApply([AdaptiveHistogramEqualization()], p=0.3),
        RandomGammaCorrection(gamma_range=(0.7, 1.3)),
        transforms.RandomApply([RandomSharpness()], p=0.3),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )
    ]
    
    # Additional transforms for minority class
    if is_minority_class:
        minority_transforms = [
            transforms.RandomApply([
                transforms.RandomResizedCrop(
                    input_size,
                    scale=(0.7, 1.0),
                    ratio=(0.8, 1.2)
                )
            ], p=0.7),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=45,
                    translate=(0.15, 0.15),
                    scale=(0.8, 1.2),
                    shear=15
                )
            ], p=0.5)
        ]
        base_transforms.extend(minority_transforms)
    else:
        base_transforms.append(
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0))
        )
    
    # Add any additional transforms
    if additional_transforms:
        base_transforms.extend(additional_transforms)
    
    # Final transforms
    final_transforms = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomApply([CustomGaussianNoise(0., 0.01)], p=0.2)
    ]
    
    return transforms.Compose(base_transforms + final_transforms)

class BalancedAugmentationWrapper:
    """Wrapper for applying different augmentations based on class"""
    
    def __init__(
        self,
        dataset,
        minority_class_idx: int = 1,  # RG class index
        mixup_prob: float = 0.2,
        cutmix_prob: float = 0.2,
        mixup_alpha: float = 0.2
    ):
        self.dataset = dataset
        self.minority_class_idx = minority_class_idx
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.mixup_alpha = mixup_alpha
        
        # Create separate transforms for minority and majority classes
        self.minority_transform = get_enhanced_augmentation_transform(
            is_minority_class=True
        )
        self.majority_transform = get_enhanced_augmentation_transform(
            is_minority_class=False
        )
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[idx]
        
        # Apply class-specific augmentation
        if isinstance(image, Image.Image):
            if label == self.minority_class_idx:
                image = self.minority_transform(image)
            else:
                image = self.majority_transform(image)
        
        # Randomly apply mixup
        if random.random() < self.mixup_prob:
            return self._apply_mixup(image, label)
        
        # Randomly apply cutmix
        if random.random() < self.cutmix_prob:
            return self._apply_cutmix(image, label)
        
        return image, label
    
    def _apply_mixup(
        self,
        image: torch.Tensor,
        label: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup augmentation"""
        idx2 = random.randint(0, len(self.dataset) - 1)
        image2, label2 = self.dataset[idx2]
        
        if isinstance(image2, Image.Image):
            image2 = (self.minority_transform if label2 == self.minority_class_idx
                      else self.majority_transform)(image2)
        
        # Generate mixup weight
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Create mixed image
        mixed_image = lam * image + (1 - lam) * image2
        
        # Create soft label
        label_onehot = torch.zeros(2)
        label_onehot[label] = lam
        label_onehot[label2] = 1 - lam
        
        return mixed_image, label_onehot
    
    def _apply_cutmix(
        self,
        image: torch.Tensor,
        label: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cutmix augmentation"""
        idx2 = random.randint(0, len(self.dataset) - 1)
        image2, label2 = self.dataset[idx2]
        
        if isinstance(image2, Image.Image):
            image2 = (self.minority_transform if label2 == self.minority_class_idx
                      else self.majority_transform)(image2)
        
        # Generate random box
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        _, h, w = image.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        cx = random.randint(0, w - cut_w)
        cy = random.randint(0, h - cut_h)
        
        # Apply cutmix
        mixed_image = image.clone()
        mixed_image[:, cy:cy+cut_h, cx:cx+cut_w] = \
            image2[:, cy:cy+cut_h, cx:cx+cut_w]
        
        # Adjust lambda to exactly reflect pixel ratio
        lam = 1 - (cut_h * cut_w) / (h * w)
        
        # Create soft label
        label_onehot = torch.zeros(2)
        label_onehot[label] = lam
        label_onehot[label2] = 1 - lam
        
        return mixed_image, label_onehot

    def __len__(self):
        return len(self.dataset)

def apply_augmentations(dataset, config):
    """
    Apply augmentations to the dataset based on the provided configuration
    
    Args:
        dataset: The original dataset
        config: Configuration dictionary containing augmentation parameters
    
    Returns:
        Augmented dataset
    """
    return BalancedAugmentationWrapper(
        dataset,
        minority_class_idx=config.get('minority_class_idx', 1),
        mixup_prob=config.get('mixup_prob', 0.2),
        cutmix_prob=config.get('cutmix_prob', 0.2),
        mixup_alpha=config.get('mixup_alpha', 0.2)
    )

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # This section can be used for testing the augmentations
    logger.info("Testing augmentations...")
    
    # Create a dummy dataset for testing
    class DummyDataset:
        def __init__(self, size=100):
            self.size = size
        
        def __getitem__(self, idx):
            # Create a dummy image and label
            image = Image.new('RGB', (256, 256), color=(73, 109, 137))
            label = random.randint(0, 1)
            return image, label
        
        def __len__(self):
            return self.size

    # Create a dummy dataset and apply augmentations
    dummy_dataset = DummyDataset()
    config = {
        'minority_class_idx': 1,
        'mixup_prob': 0.3,
        'cutmix_prob': 0.3,
        'mixup_alpha': 0.2
    }
    augmented_dataset = apply_augmentations(dummy_dataset, config)

    # Test a few augmented samples
    for i in range(5):
        augmented_image, augmented_label = augmented_dataset[i]
        logger.info(f"Sample {i}: Shape: {augmented_image.shape}, Label: {augmented_label}")

    logger.info("Augmentation testing completed.")