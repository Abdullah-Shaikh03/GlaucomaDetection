import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class ModelVisualizer:
    def __init__(self, analysis_results_path):
        self.results = np.load(analysis_results_path)
    
    def plot_training_history(self, history_file, output_path='analysis_results/training_history.png'):
        """Plot training history."""
        history = np.load(history_file, allow_pickle=True).item()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot accuracy
        ax1.plot(history['train_acc'], label='Train Accuracy')
        ax1.plot(history['val_acc'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history['train_loss'], label='Train Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_confusion_matrix(self, output_path='analysis_results/confusion_matrix.png'):
        """Plot confusion matrix."""
        cm = self.results['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(output_path)
        plt.close()

    def plot_feature_maps(self, output_path='analysis_results/feature_maps.png'):
        """Plot feature maps."""
        feature_maps = self.results['feature_maps'][0]  # First sample
        n_features = min(16, feature_maps.shape[0])
        
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        for idx in range(n_features):
            i, j = idx // 4, idx % 4
            axs[i, j].imshow(feature_maps[idx], cmap='viridis')
            axs[i, j].axis('off')
        
        plt.suptitle('Feature Maps from First Convolutional Layer')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def print_metrics(self):
        """Print test metrics."""
        print(f"Test Accuracy: {self.results['test_accuracy']:.2f}%")
        print(f"Test Loss: {self.results['test_loss']:.4f}")

def main():
    # Example usage
    visualizer = ModelVisualizer('models/analysis_results.npz')
    
    # Generate all visualizations
    visualizer.plot_training_history('models/training_history.npy')
    visualizer.plot_confusion_matrix()
    visualizer.plot_feature_maps()
    visualizer.print_metrics()

if __name__ == "__main__":
    main()