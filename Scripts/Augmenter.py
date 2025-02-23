import torch
import random
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from torchvision import transforms
from typing import Tuple, Optional, List
import logging

class GlaucomaSpecificAugmentation:
    """Custom augmentation class for glaucoma-specific image enhancements"""
    def __init__(self):
        self.vessel_enhance = transforms.RandomApply([
            ImageFilter.UnsharpMask(radius=2, percent=150)
        ], p=0.4)
        
        self.contrast_enhance = transforms.RandomApply([
            lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(1.2, 1.5))
        ], p=0.5)

    def __call__(self, img):
        img = self.vessel_enhance(img)
        img = self.contrast_enhance(img)
        return img

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

class AugmentationMonitor:
    """Monitor and track augmentation statistics"""
    def __init__(self):
        self.aug_stats = {
            'mixup_applied': 0,
            'cutmix_applied': 0,
            'vessel_enhancement_applied': 0
        }
    
    def update(self, aug_type):
        self.aug_stats[aug_type] += 1
    
    def get_stats(self):
        total = sum(self.aug_stats.values())
        return {k: v/total for k, v in self.aug_stats.items()}

def get_enhanced_augmentation_transform(
    is_minority_class: bool = False,
    input_size: int = 256,
    additional_transforms: Optional[List] = None
) -> transforms.Compose:
    """
    Get enhanced augmentation transforms optimized for glaucoma detection
    
    Args:
        is_minority_class: Whether these transforms are for the minority class
        input_size: Input image size
        additional_transforms: Additional transform operations to include
    
    Returns:
        Composed transformation pipeline
    """
    base_transforms = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15, fill=0),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.3),
        transforms.RandomApply([AdaptiveHistogramEqualization()], p=0.4),
        RandomGammaCorrection(gamma_range=(0.8, 1.2)),
        transforms.RandomApply([RandomSharpness(
            sharpness_range=(0.9, 1.3)
        )], p=0.4),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.1,
            hue=0.05
        )
    ]
    
    if is_minority_class:
        minority_transforms = [
            transforms.RandomApply([
                transforms.RandomResizedCrop(
                    input_size,
                    scale=(0.85, 1.0),
                    ratio=(0.9, 1.1)
                )
            ], p=0.6),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=30,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=10
                )
            ], p=0.4)
        ]
        base_transforms.extend(minority_transforms)
    else:
        base_transforms.append(
            transforms.RandomResizedCrop(
                input_size,
                scale=(0.9, 1.0),
                ratio=(0.95, 1.05)
            )
        )
    
    if additional_transforms:
        base_transforms.extend(additional_transforms)
    
    final_transforms = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.3460874557495117, 0.22415442764759064, 0.14760617911815643],
            std=[0.22597847878932953, 0.15277379751205444, 0.10380984842777252]
        ),
        transforms.RandomApply([CustomGaussianNoise(0., 0.005)], p=0.15)
    ]
    
    return transforms.Compose(base_transforms + final_transforms)

class BalancedAugmentationWrapper:
    """Wrapper for applying different augmentations based on class"""
    
    def __init__(
        self,
        dataset,
        minority_class_idx: int = 1,
        mixup_prob: float = 0.2,
        cutmix_prob: float = 0.2,
        mixup_alpha: float = 0.2
    ):
        self.dataset = dataset
        self.minority_class_idx = minority_class_idx
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.mixup_alpha = mixup_alpha
        self.monitor = AugmentationMonitor()
        
        self.minority_transform = get_enhanced_augmentation_transform(
            is_minority_class=True
        )
        self.majority_transform = get_enhanced_augmentation_transform(
            is_minority_class=False
        )
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[idx]
        
        if isinstance(image, Image.Image):
            if label == self.minority_class_idx:
                image = self.minority_transform(image)
                self.monitor.update('vessel_enhancement_applied')
            else:
                image = self.majority_transform(image)
        
        if random.random() < self.mixup_prob:
            self.monitor.update('mixup_applied')
            return self._apply_mixup(image, label)
        
        if random.random() < self.cutmix_prob:
            self.monitor.update('cutmix_applied')
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
        
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        mixed_image = lam * image + (1 - lam) * image2
        
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
        
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        _, h, w = image.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        cx = random.randint(0, w - cut_w)
        cy = random.randint(0, h - cut_h)
        
        mixed_image = image.clone()
        mixed_image[:, cy:cy+cut_h, cx:cx+cut_w] = \
            image2[:, cy:cy+cut_h, cx:cx+cut_w]
        
        lam = 1 - (cut_h * cut_w) / (h * w)
        
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
    logger.info("Testing augmentations...")
    
    # Create a dummy dataset for testing
    class DummyDataset:
        def __init__(self, size=100):
            self.size = size
        
        def __getitem__(self, idx):
            image = Image.new('RGB', (256, 256), color=(73, 109, 137))
            label = random.randint(0, 1)
            return image, label
        
        def __len__(self):
            return self.size

    # Test the augmentations
    config = {
        'minority_class_idx': 1,
        'mixup_prob': 0.3,
        'cutmix_prob': 0.3,
        'mixup_alpha': 0.2
    }
    
    dummy_dataset = DummyDataset()
    augmented_dataset = apply_augmentations(dummy_dataset, config)

    # Test samples
    for i in range(5):
        augmented_image, augmented_label = augmented_dataset[i]
        logger.info(f"Sample {i}: Shape: {augmented_image.shape}, Label: {augmented_label}")
        
    # Print augmentation statistics
    logger.info("Augmentation statistics:")
    logger.info(augmented_dataset.monitor.get_stats())
    
    logger.info("Augmentation testing completed.")