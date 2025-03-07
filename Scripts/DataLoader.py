from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import os
import torch
from typing import Tuple, List
import logging
from pathlib import Path
from PIL import Image

class GlaucomaDataset(Dataset):
    """Enhanced GlaucomaDataset with class balancing and error handling"""
    
    def __init__(self, root_dir: str, transform=None, mode: str = 'train'):
        self.root_dir = Path(root_dir) / mode
        self.transform = transform
        self.mode = mode
        self.classes = ['NRG', 'RG']  # Non-Referrable Glaucoma, Referrable Glaucoma
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        self.class_counts = {class_name: 0 for class_name in self.classes}
        
        self._load_dataset()
        self._compute_statistics()
        
    def _load_dataset(self):
        """Load dataset and compute class distributions"""
        for class_idx, class_name in enumerate(self.classes):
            class_path = self.root_dir / class_name
            if not class_path.exists():
                logging.warning(f"Class directory not found: {class_path}")
                continue  # Skip missing classes
                
            for entry in os.scandir(class_path):
                if entry.is_file() and entry.name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(str(entry.path))
                    self.labels.append(class_idx)
                    self.class_counts[class_name] += 1
    
    def _compute_statistics(self):
        """Compute dataset statistics"""
        total_samples = len(self.labels)
        if total_samples == 0:
            logging.warning(f"No samples found for {self.mode} dataset!")
            self.class_weights = {class_name: 1.0 for class_name in self.classes}
            return
            
        self.class_weights = {
            class_name: total_samples / (len(self.classes) * count)
            for class_name, count in self.class_counts.items() if count > 0
        }
        
        # Ensure all classes have weights (use 1.0 for empty classes)
        for class_name in self.classes:
            if class_name not in self.class_weights:
                self.class_weights[class_name] = 1.0
        
        logging.info(f"{self.mode} dataset statistics:")
        logging.info(f"Total samples: {total_samples}")
        for class_name, count in self.class_counts.items():
            if count > 0:
                percentage = (count / total_samples) * 100
                logging.info(f"{class_name}: {count} ({percentage:.2f}%) - Weight: {self.class_weights[class_name]:.2f}")

    def get_sample_weights(self) -> torch.Tensor:
        """Generate sample weights for WeightedRandomSampler"""
        weights = [self.class_weights[self.classes[label]] for label in self.labels]
        return torch.DoubleTensor(weights)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load directly as PIL Image - this fixes the transform issue
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {str(e)}")
            image = Image.new('RGB', (256, 256), color=0)  # Black PIL Image as fallback
        
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                logging.error(f"Error applying transform to {img_path}: {str(e)}")
                # Return zeros tensor with expected dimensions after transform
                image = torch.zeros((3, 256, 256))
        
        return image, label


def get_transforms(mode: str = 'train', input_size: int = 256) -> transforms.Compose:
    """Get transforms for different dataset modes"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize
        ])


def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    input_size: int = 256,
    num_workers: int = 4,
    use_balanced_sampling: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders with optional balanced sampling for training
    
    Args:
        data_dir: Root directory containing the dataset
        batch_size: Batch size for the data loaders
        input_size: Input image size
        num_workers: Number of worker processes for data loading
        use_balanced_sampling: Whether to use balanced sampling for training
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets with appropriate transforms
    train_dataset = GlaucomaDataset(data_dir, transform=get_transforms('train', input_size), mode='train')
    val_dataset = GlaucomaDataset(data_dir, transform=get_transforms('val', input_size), mode='validation')
    test_dataset = GlaucomaDataset(data_dir, transform=get_transforms('test', input_size), mode='test')

    # Configure training sampler
    train_sampler = None
    if use_balanced_sampling and len(train_dataset) > 0:
        train_sampler = WeightedRandomSampler(
            weights=train_dataset.get_sample_weights(),
            num_samples=len(train_dataset),
            replacement=True
        )

    # Create data loaders with safety checks
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler, 
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir='path/to/dataset',
        batch_size=32,
        input_size=256,
        num_workers=4,
        use_balanced_sampling=True
    )
    
    # Print dataset sizes
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Example of iterating through a batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        break  # Just show the first batch