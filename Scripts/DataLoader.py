from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import os
import torch
from typing import Tuple, List, Union, Dict, Optional
import logging
from pathlib import Path
import zipfile
import tarfile
import io
import tempfile
from contextlib import contextmanager
import numpy as np

# Optional py7zr import
try:
    import py7zr
    SEVEN_ZIP_AVAILABLE = True
except ImportError:
    SEVEN_ZIP_AVAILABLE = False
    logging.warning("py7zr not installed. 7-Zip support will be disabled. Install with: pip install py7zr")

def get_transforms(mode: str = 'train', input_size: int = 256) -> transforms.Compose:
    """
    Get transforms for different dataset modes
    
    Args:
        mode: Dataset mode ('train', 'val', or 'test')
        input_size: Input image size
        
    Returns:
        Composition of transforms
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
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

class CompressedFileHandler:
    """Handles different types of compressed files"""
    
    @staticmethod
    @contextmanager
    def get_file_handler(file_path: Union[str, Path]) -> Optional[Union[zipfile.ZipFile, tarfile.TarFile, 'py7zr.SevenZipFile']]:
        """Context manager to handle different compression formats"""
        file_path = str(file_path)
        try:
            if zipfile.is_zipfile(file_path):
                with zipfile.ZipFile(file_path, 'r') as z:
                    yield z
            elif tarfile.is_tarfile(file_path):
                with tarfile.open(file_path, 'r:*') as t:
                    yield t
            elif file_path.endswith('.7z') and SEVEN_ZIP_AVAILABLE:
                with py7zr.SevenZipFile(file_path, 'r') as sz:
                    yield sz
            else:
                yield None
        except Exception as e:
            logging.error(f"Error opening compressed file {file_path}: {str(e)}")
            yield None

class GlaucomaDataset(Dataset):
    """Enhanced GlaucomaDataset with support for compressed formats and modified path structure"""
    
    def __init__(self, 
                 root_path: Union[str, Path], 
                 transform=None, 
                 mode: str = 'train',
                 data_subdir: str = 'raw'):
        """
        Initialize dataset
        
        Args:
            root_path: Path to dataset directory or compressed file
            transform: Optional transforms to apply to images
            mode: Dataset split ('train', 'validation', or 'test')
            data_subdir: Subdirectory containing the data splits (default: 'raw')
        """
        self.root_path = Path(root_path)
        self.transform = transform
        self.mode = mode
        self.data_subdir = data_subdir
        self.classes = ['NRG', 'RG']
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        self.class_counts = {class_name: 0 for class_name in self.classes}
        self.is_compressed = self._check_if_compressed()
        
        self._load_dataset()
        self._compute_statistics()
    
    def _check_if_compressed(self) -> bool:
        """Check if the root path is a compressed file"""
        root_str = str(self.root_path)
        return any([
            zipfile.is_zipfile(root_str),
            tarfile.is_tarfile(root_str),
            root_str.endswith('.7z') and SEVEN_ZIP_AVAILABLE
        ])
    
    def _load_dataset(self):
        """Load dataset from either compressed file or directory"""
        if self.is_compressed:
            self._load_from_compressed()
        else:
            self._load_from_directory()
    
    def _load_from_compressed(self):
        """Load dataset from a compressed file"""
        with CompressedFileHandler.get_file_handler(self.root_path) as archive:
            if archive is None:
                raise ValueError(f"Unable to open compressed file: {self.root_path}")
            
            # List all files in archive
            if isinstance(archive, zipfile.ZipFile):
                all_files = archive.namelist()
            elif isinstance(archive, tarfile.TarFile):
                all_files = archive.getnames()
            elif SEVEN_ZIP_AVAILABLE and isinstance(archive, py7zr.SevenZipFile):
                all_files = archive.getnames()
            
            # Filter files for current mode
            mode_path = f"{self.data_subdir}/{self.mode}"
            for file_path in all_files:
                if mode_path in file_path and any(ext in file_path.lower() for ext in ['.jpg', '.jpeg', '.png']):
                    label = 1 if 'RG' in file_path else 0
                    self.image_paths.append(file_path)
                    self.labels.append(label)
                    self.class_counts[self.classes[label]] += 1
    
    def _load_from_directory(self):
        """Load dataset from a directory"""
        mode_path = self.root_path / self.data_subdir / self.mode
        for class_idx, class_name in enumerate(self.classes):
            class_path = mode_path / class_name
            if not class_path.exists():
                continue
                
            for img_path in class_path.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.image_paths.append(str(img_path))
                    self.labels.append(class_idx)
                    self.class_counts[class_name] += 1
    
    def _compute_statistics(self):
        """Compute dataset statistics"""
        self.total_samples = len(self.image_paths)
        self.class_weights = [
            len(self.image_paths) / (len(self.classes) * count) 
            if count > 0 else 0 
            for count in self.class_counts.values()
        ]
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get sample weights for balanced sampling"""
        return torch.tensor([self.class_weights[label] for label in self.labels])
    
    def _load_image(self, path: str) -> Image.Image:
        """Load image from either compressed file or directory"""
        if self.is_compressed:
            with CompressedFileHandler.get_file_handler(self.root_path) as archive:
                if isinstance(archive, zipfile.ZipFile):
                    with archive.open(path) as f:
                        img_data = f.read()
                elif isinstance(archive, tarfile.TarFile):
                    img_data = archive.extractfile(path).read()
                elif SEVEN_ZIP_AVAILABLE and isinstance(archive, py7zr.SevenZipFile):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        archive.extract(path=tmpdir, targets=[path])
                        img_data = (Path(tmpdir) / path).read_bytes()
                return Image.open(io.BytesIO(img_data)).convert('RGB')
        else:
            return Image.open(path).convert('RGB')
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset"""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = self._load_image(img_path)
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    input_size: int = 256,
    num_workers: int = 2,
    use_balanced_sampling: bool = True,
    data_subdir: str = 'raw'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders with support for the new directory structure
    
    Args:
        data_dir: Root directory or compressed file containing the dataset
        batch_size: Batch size for the data loaders
        input_size: Input image size
        num_workers: Number of worker processes for data loading
        use_balanced_sampling: Whether to use balanced sampling for training
        data_subdir: Subdirectory containing the data splits (default: 'raw')
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = GlaucomaDataset(
        data_dir,
        transform=get_transforms('train', input_size),
        mode='train',
        data_subdir=data_subdir
    )
    
    val_dataset = GlaucomaDataset(
        data_dir,
        transform=get_transforms('val', input_size),
        mode='validation',
        data_subdir=data_subdir
    )
    
    test_dataset = GlaucomaDataset(
        data_dir,
        transform=get_transforms('test', input_size),
        mode='test',
        data_subdir=data_subdir
    )

    if use_balanced_sampling:
        train_sampler = WeightedRandomSampler(
            weights=train_dataset.get_sample_weights(),
            num_samples=len(train_dataset),
            replacement=True
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
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