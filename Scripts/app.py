import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class FundusDataset(Dataset):
    """Custom dataset for FUNDUS images to enable efficient GPU processing"""
    def __init__(self, root_dir, split, cls, image_extension='.jpg'):
        self.image_dir = os.path.join(root_dir, split, cls)
        self.image_extension = image_extension
        self.image_files = []
        
        if os.path.exists(self.image_dir):
            self.image_files = [f for f in os.listdir(self.image_dir) 
                               if f.lower().endswith(self.image_extension)]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            # Return a placeholder if image can't be read
            return np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize to 0-1 range
        img_tensor = torch.from_numpy(img).float() / 255.0
        # Reorder dims to [C, H, W] format for PyTorch
        img_tensor = img_tensor.permute(2, 0, 1)
        
        return img_tensor

def calculate_dataset_stats_gpu(dataset_path, image_extension='.jpg', sample_size=None, 
                           split_stats=True, class_stats=True, batch_size=16):
    """
    Calculate mean and standard deviation of images in the dataset using GPU acceleration.
    """
    stats_dict = {}
    splits = ['train', 'test', 'validation']
    classes = ['NRG', 'RG']
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize arrays for collecting all means and stds
    all_means = []
    all_stds = []
    
    # Process each split and class
    for split in splits:
        if split_stats:
            split_means = []
            split_stds = []
            
        for cls in classes:
            # Create dataset
            dataset = FundusDataset(dataset_path, split, cls, image_extension)
            
            # Skip if empty
            if len(dataset) == 0:
                print(f"Warning: No images found in {split}/{cls}, skipping.")
                continue
                
            # Sample if specified
            if sample_size and sample_size < len(dataset):
                indices = torch.randperm(len(dataset))[:sample_size]
                dataset = torch.utils.data.Subset(dataset, indices)
            
            # Create dataloader
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
            
            # Initialize arrays for this class
            if class_stats:
                class_means = []
                class_stds = []
            
            print(f"Processing {len(dataset)} images in {split}/{cls}...")
            
            # Process batches
            for batch in tqdm(dataloader):
                batch = batch.to(device)
                
                # Calculate mean and std for each image in batch
                batch_mean = torch.mean(batch, dim=[2, 3])
                batch_std = torch.std(batch, dim=[2, 3])
                
                # Move back to CPU for numpy operations
                batch_mean = batch_mean.cpu().numpy()
                batch_std = batch_std.cpu().numpy()
                
                # Append to appropriate lists
                all_means.extend(batch_mean)
                all_stds.extend(batch_std)
                
                if split_stats:
                    split_means.extend(batch_mean)
                    split_stds.extend(batch_std)
                
                if class_stats:
                    class_means.extend(batch_mean)
                    class_stds.extend(batch_std)
            
            # Calculate class statistics if requested
            if class_stats and len(class_means) > 0:
                class_means = np.array(class_means)
                class_stds = np.array(class_stds)
                class_overall_mean = np.mean(class_means, axis=0)
                class_overall_std = np.mean(class_stds, axis=0)
                stats_dict[f"{split}_{cls}_mean"] = class_overall_mean
                stats_dict[f"{split}_{cls}_std"] = class_overall_std
        
        # Calculate split statistics if requested
        if split_stats and len(split_means) > 0:
            split_means = np.array(split_means)
            split_stds = np.array(split_stds)
            split_overall_mean = np.mean(split_means, axis=0)
            split_overall_std = np.mean(split_stds, axis=0)
            stats_dict[f"{split}_mean"] = split_overall_mean
            stats_dict[f"{split}_std"] = split_overall_std
    
    # Calculate overall statistics
    if len(all_means) > 0:
        all_means = np.array(all_means)
        all_stds = np.array(all_stds)
        overall_mean = np.mean(all_means, axis=0)
        overall_std = np.mean(all_stds, axis=0)
        stats_dict["overall_mean"] = overall_mean
        stats_dict["overall_std"] = overall_std
    
    # Scale back to 0-255 range
    for key in stats_dict:
        stats_dict[key] = stats_dict[key] * 255
    
    return stats_dict

def plot_class_histograms_gpu(dataset_path, image_extension='.jpg', sample_size=5):
    """
    Plot histograms of pixel values using GPU acceleration.
    """
    splits = ['train']  # Use only training data for histograms
    classes = ['NRG', 'RG']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    fig, axs = plt.subplots(len(classes), 3, figsize=(15, 5*len(classes)))
    
    for i, cls in enumerate(classes):
        # Create dataset
        dataset = FundusDataset(dataset_path, splits[0], cls, image_extension)
        
        if len(dataset) == 0:
            print(f"Warning: No images found in {splits[0]}/{cls}, skipping.")
            continue
            
        # Sample if specified
        if sample_size and sample_size < len(dataset):
            indices = torch.randperm(len(dataset))[:sample_size]
            sampled_indices = indices.tolist()
        else:
            sampled_indices = range(min(len(dataset), sample_size))
        
        # Combine pixel values from all sample images
        r_values = []
        g_values = []
        b_values = []
        
        for idx in sampled_indices:
            img_tensor = dataset[idx].to(device)
            
            # Move to CPU for histogram creation
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            
            # Collect pixel values
            r_values.extend(img_np[0].flatten())
            g_values.extend(img_np[1].flatten())
            b_values.extend(img_np[2].flatten())
        
        # Plot histograms
        axs[i, 0].hist(r_values, bins=50, color='red', alpha=0.7)
        axs[i, 0].set_title(f"Red Channel - {cls}")
        axs[i, 0].set_xlabel('Pixel Value')
        axs[i, 0].set_ylabel('Frequency')
        
        axs[i, 1].hist(g_values, bins=50, color='green', alpha=0.7)
        axs[i, 1].set_title(f"Green Channel - {cls}")
        axs[i, 1].set_xlabel('Pixel Value')
        
        axs[i, 2].hist(b_values, bins=50, color='blue', alpha=0.7)
        axs[i, 2].set_title(f"Blue Channel - {cls}")
        axs[i, 2].set_xlabel('Pixel Value')
    
    plt.tight_layout()
    plt.savefig('class_histograms.png')
    plt.close()

def count_images(dataset_path, image_extension='.jpg'):
    """
    Count the number of images in each split and class.
    (No GPU acceleration needed for this function)
    """
    splits = ['train', 'test', 'validation']
    classes = ['NRG', 'RG']
    
    counts = []
    
    for split in splits:
        for cls in classes:
            class_dir = os.path.join(dataset_path, split, cls)
            if not os.path.exists(class_dir):
                count = 0
            else:
                count = len([f for f in os.listdir(class_dir) 
                            if f.lower().endswith(image_extension)])
            
            counts.append({
                'Split': split,
                'Class': cls,
                'Count': count
            })
    
    # Create DataFrame
    counts_df = pd.DataFrame(counts)
    
    # Add totals
    split_totals = counts_df.groupby('Split')['Count'].sum().reset_index()
    split_totals['Class'] = 'Total'
    
    class_totals = counts_df.groupby('Class')['Count'].sum().reset_index()
    class_totals['Split'] = 'Total'
    
    total = counts_df['Count'].sum()
    overall_total = pd.DataFrame([{'Split': 'Total', 'Class': 'Total', 'Count': total}])
    
    counts_df = pd.concat([counts_df, split_totals, class_totals, overall_total])
    
    return counts_df

def main():
    parser = argparse.ArgumentParser(description='Calculate mean and std for hierarchical FUNDUS dataset using GPU')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset root directory')
    parser.add_argument('--extension', type=str, default='.jpg', help='Image file extension')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of images to sample per class')
    parser.add_argument('--plot_histograms', action='store_true', help='Plot histograms of sample images')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("No GPU detected, falling back to CPU.")
    
    # Count images in dataset
    print("Counting images in dataset...")
    counts_df = count_images(args.dataset_path, args.extension)
    print("\nImage counts:")
    print(counts_df.to_string(index=False))
    
    # Calculate statistics using GPU
    print("\nCalculating statistics using GPU acceleration...")
    stats_dict = calculate_dataset_stats_gpu(args.dataset_path, args.extension, args.sample_size, 
                                           batch_size=args.batch_size)
    
    # Print results
    print("\nDataset Statistics:")
    for key, value in stats_dict.items():
        print(f"{key}: {value}")
    
    # Save results to file
    with open('dataset_stats.txt', 'w') as f:
        f.write(f"Dataset: {args.dataset_path}\n\n")
        f.write("Image counts:\n")
        f.write(counts_df.to_string(index=False))
        f.write("\n\nDataset Statistics:\n")
        for key, value in stats_dict.items():
            f.write(f"{key}: {value}\n")
    
    # Create a structured DataFrame for statistics
    stats_rows = []
    for key, value in stats_dict.items():
        parts = key.split('_')
        if len(parts) == 3:  # split_class_stat
            split, cls, stat = parts
        elif len(parts) == 2:  # split_stat or overall_stat
            if parts[0] == 'overall':
                split = 'Overall'
                cls = 'All'
                stat = parts[1]
            else:
                split = parts[0]
                cls = 'All'
                stat = parts[1]
        
        stats_rows.append({
            'Split': split,
            'Class': cls,
            'Statistic': stat,
            'R': value[0],
            'G': value[1],
            'B': value[2]
        })
    
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv('dataset_stats.csv', index=False)
    print("\nDetailed statistics saved to 'dataset_stats.csv'")
    
    # Plot histograms if requested
    if args.plot_histograms:
        print("Generating histograms using GPU acceleration...")
        plot_class_histograms_gpu(args.dataset_path, args.extension, args.sample_size)
        print("Class histograms saved to 'class_histograms.png'")

if __name__ == "__main__":
    main()