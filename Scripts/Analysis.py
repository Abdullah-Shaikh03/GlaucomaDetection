import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from Model import get_model
from DataLoader import GlaucomaDataset, get_data_loaders
from torchvision import transforms
import json
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelAnalyzer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def compute_confusion_matrix(self, test_loader):
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return confusion_matrix(all_labels, all_preds), all_labels, all_preds

    def extract_feature_maps(self, sample_input):
        features = None
        
        def hook(module, input, output):
            nonlocal features
            features = output.detach().cpu()
        
        hook_handle = self.model.conv1.register_forward_hook(hook)
        
        with torch.no_grad():
            self.model(sample_input)
        hook_handle.remove()
        
        return features

    def compute_metrics(self, test_loader):
        correct = 0
        total = 0
        total_loss = 0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(test_loader)
        return accuracy, avg_loss

def load_model_and_metadata(checkpoint_path, metadata_path):
    try:
        checkpoint = torch.load(checkpoint_path)
        print("Checkpoint contents:", checkpoint.keys())
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        checkpoint = None

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print("Metadata contents:", metadata.keys())
    except Exception as e:
        print(f"Error loading metadata: {str(e)}")
        metadata = None

    return checkpoint, metadata

def save_analysis_results(analyzer, test_loader, sample_input, training_history, checkpoint, metadata, output_path='analysis_results.npz', results_file='AnalysisResults.txt'):
    print("Computing confusion matrix...")
    cm, true_labels, pred_labels = analyzer.compute_confusion_matrix(test_loader)
    
    print("Extracting feature maps...")
    feature_maps = analyzer.extract_feature_maps(sample_input)
    
    print("Computing metrics...")
    accuracy, loss = analyzer.compute_metrics(test_loader)
    
    print("Saving results...")
    np.savez(output_path, 
             confusion_matrix=cm,
             feature_maps=feature_maps.numpy(),
             test_accuracy=accuracy,
             test_loss=loss,
             training_history=training_history)
    
    print(f"Analysis results saved to {output_path}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Loss: {loss:.4f}")

    # Generate classification report
    class_names = ['Non-Glaucoma', 'Glaucoma']  # Adjust if your classes are different
    report = classification_report(true_labels, pred_labels, target_names=class_names)

    # Write results to text file
    
    # Ensure directory exists before saving the results
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
    
        f.write("Glaucoma Detection Model Analysis Results\n")
        f.write("========================================\n\n")
        
        f.write("Model Information:\n")
        f.write(f"  Architecture: {type(analyzer.model).__name__}\n")
        if checkpoint and 'epoch' in checkpoint:
            f.write(f"  Trained for {checkpoint['epoch']} epochs\n")
        if metadata and 'best_val_acc' in metadata:
            f.write(f"  Best validation accuracy: {metadata['best_val_acc']:.2f}%\n")
        f.write("\n")

        f.write("Test Set Performance:\n")
        f.write(f"  Accuracy: {accuracy:.2f}%\n")
        f.write(f"  Loss: {loss:.4f}\n")
        f.write("\n")

        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n")

        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\n")

        f.write("Training History Summary:\n")
        if isinstance(training_history, dict):
            for key, value in training_history.items():
                if isinstance(value, list):
                    f.write(f"  Final {key}: {value[-1]:.4f}\n")

    print(f"Detailed analysis results saved to {results_file}")

def main():
    # Device configuration
    print(f"Using device: {device}")
    
    try:
        # Load model and metadata
        checkpoint_path = 'models/model_state.pth'
        metadata_path = 'models/metadata.json'
        checkpoint, metadata = load_model_and_metadata(checkpoint_path, metadata_path)
        
        if checkpoint is None or metadata is None:
            raise ValueError("Failed to load checkpoint or metadata")

        # Load training history
        training_history = np.load('models/training_history.npy', allow_pickle=True).item()
        
        # Initialize model
        model = get_model(device)
        
        # Load trained model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully")
        
        # Print some information about the loaded model
        if 'epoch' in checkpoint:
            print(f"Loaded model from epoch {checkpoint['epoch']}")
        if 'best_val_acc' in checkpoint:
            print(f"Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
        
        # Get data loaders
        data_dir = './raw/'  # Update this to your actual data directory
        _, _, test_loader = get_data_loaders(data_dir, batch_size=32)
        print("Test dataset loaded successfully")
        
        # Initialize analyzer
        analyzer = ModelAnalyzer(model, device)
        
        # Get a sample input
        test_dataset = test_loader.dataset
        sample_input, _ = test_dataset[0]
        sample_input = sample_input.unsqueeze(0).to(device)
        
        # Run analysis
        save_analysis_results(analyzer, test_loader, sample_input, training_history, checkpoint, metadata, 
                              output_path='models/analysis_results.npz', 
                              results_file='analysis_results/AnalysisResults.txt')
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()