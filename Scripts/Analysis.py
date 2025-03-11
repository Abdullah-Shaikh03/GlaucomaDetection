import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms
from Model import get_model
from DataLoader import GlaucomaDataset, get_data_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelAnalyzer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def compute_confusion_matrix(self, test_loader):
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return confusion_matrix(all_labels, all_preds), all_labels, all_preds

    def extract_feature_maps(self, sample_input):
        feature_maps = {}
        
        def hook(module, input, output, name):
            feature_maps[name] = output.detach().cpu()
        
        hooks = []
        for name, layer in self.model.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                hooks.append(layer.register_forward_hook(lambda m, i, o, n=name: hook(m, i, o, n)))
        
        with torch.no_grad():
            self.model(sample_input)
        
        for h in hooks:
            h.remove()
        
        return feature_maps

    def compute_metrics(self, test_loader):
        correct, total, total_loss = 0, 0, 0
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
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        checkpoint = None
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
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
             feature_maps={k: v.numpy() for k, v in feature_maps.items()},
             test_accuracy=accuracy,
             test_loss=loss,
             training_history=training_history)
    
    print(f"Analysis results saved to {output_path}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Loss: {loss:.4f}")
    
    class_names = metadata.get('class_names', [str(i) for i in range(len(set(true_labels)))])
    report = classification_report(true_labels, pred_labels, target_names=class_names)
    
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        f.write("Model Analysis Results\n")
        f.write("======================\n\n")
        f.write(f"Model Architecture: {type(analyzer.model).__name__}\n")
        if checkpoint and 'epoch' in checkpoint:
            f.write(f"Trained for {checkpoint['epoch']} epochs\n")
        if metadata and 'best_val_acc' in metadata:
            f.write(f"Best validation accuracy: {metadata['best_val_acc']:.2f}%\n")
        f.write(f"\nTest Accuracy: {accuracy:.2f}%\n")
        f.write(f"Test Loss: {loss:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        if isinstance(training_history, dict):
            for key, value in training_history.items():
                if isinstance(value, list):
                    f.write(f"Final {key}: {value[-1]:.4f}\n")
    print(f"Detailed analysis results saved to {results_file}")

def main():
    print(f"Using device: {device}")
    
    checkpoint_path, metadata_path = 'models/model_state.pth', 'models/metadata.json'
    checkpoint, metadata = load_model_and_metadata(checkpoint_path, metadata_path)
    if not checkpoint or not metadata:
        raise ValueError("Failed to load checkpoint or metadata")
    
    training_history = np.load('models/training_history.npy', allow_pickle=True).item()
    model = get_model(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    
    data_dir = './raw/'
    _, _, test_loader = get_data_loaders(data_dir, batch_size=32)
    print("Test dataset loaded successfully")
    
    analyzer = ModelAnalyzer(model, device)
    test_dataset = test_loader.dataset
    sample_input, _ = test_dataset[0]
    sample_input = sample_input.unsqueeze(0).to(device)
    
    save_analysis_results(analyzer, test_loader, sample_input, training_history, checkpoint, metadata, 
                          output_path='models/analysis_results.npz', 
                          results_file='analysis_results/AnalysisResults.txt')
    
if __name__ == "__main__":
    main()
