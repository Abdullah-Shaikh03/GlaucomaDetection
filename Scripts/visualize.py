import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import torch
from torchvision.utils import make_grid
 # Load model and extract feature maps
from Model import get_model  # Ensure correct import
import torch
    

class ModelVisualizer:
    def __init__(self, analysis_results_path, metadata_path):
        self.results = np.load(analysis_results_path, allow_pickle=True)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.class_names = self.metadata.get('class_names', ['Class 0', 'Class 1'])
    
    def plot_training_history(self, history_file, output_path='analysis_results/training_history.png'):
        """Plot training history."""
        history = np.load(history_file, allow_pickle=True).item()
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot accuracy
        if 'train_acc' in history and 'val_acc' in history:
            axes[0].plot(history['train_acc'], label='Train Accuracy')
            axes[0].plot(history['val_acc'], label='Validation Accuracy')
            axes[0].set_title('Model Accuracy')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_xlabel('Epoch')
            axes[0].legend()
        
        # Plot loss
        if 'train_loss' in history and 'val_loss' in history:
            axes[1].plot(history['train_loss'], label='Train Loss')
            axes[1].plot(history['val_loss'], label='Validation Loss')
            axes[1].set_title('Model Loss')
            axes[1].set_ylabel('Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_confusion_matrix(self, output_path='analysis_results/confusion_matrix.png'):
        """Plot confusion matrix."""
        cm = self.results['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(output_path)
        plt.close()

    def plot_feature_maps(self, model, sample_input, output_path='analysis_results/feature_maps.png'):
        """Plot feature maps from the first convolutional layer."""
        feature_maps = None
    
        def hook(module, input, output):
            nonlocal feature_maps
            feature_maps = output.detach().cpu()
    
    # Identify first convolutional layer dynamically
        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                hook_handle = layer.register_forward_hook(hook)
                break
    
        with torch.no_grad():
            model(sample_input)
        hook_handle.remove()
    
        if feature_maps is not None:
            n_features = min(16, feature_maps.shape[1])
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))  # 4x4 grid for 16 feature maps
            axes = axes.flatten()
    
            for i in range(n_features):
                axes[i].imshow(feature_maps[0, i].numpy(), cmap='gray')
                axes[i].axis('off')
    
            plt.suptitle('Feature Maps from First Convolutional Layer')
            plt.savefig(output_path)
            plt.close()
    

    def print_metrics(self):
        """Print test metrics."""
        print(f"Test Accuracy: {self.results['test_accuracy']:.2f}%")
        print(f"Test Loss: {self.results['test_loss']:.4f}")


def main():
    # Example usage
    metadata_path = 'models/metadata.json'
    visualizer = ModelVisualizer('models/analysis_results.npz', metadata_path)
    
    # Generate all visualizations
    visualizer.plot_training_history('models/training_history.npy')
    visualizer.plot_confusion_matrix()
    visualizer.print_metrics()
    
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device)
    model.load_state_dict(torch.load('models/model_state.pth', map_location=device, weights_only=False)['model_state_dict'])
    model.eval()
    
    # Get a sample input
    sample_input = torch.rand(1, 3, 224, 224).to(device)  # Modify shape as per dataset
    visualizer.plot_feature_maps(model, sample_input)

if __name__ == "__main__":
    main()
