import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
from typing import Optional, Dict, Any, Union
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
import shap

class EnhancedVisualizer:
    """Enhanced visualization class with additional features and error handling"""
    
    def __init__(self, output_dir: Union[str, Path], tensorboard_dir: Optional[str] = None):
        """
        Initialize the visualizer with output directory and optional tensorboard support
        
        Args:
            output_dir: Directory to save visualizations
            tensorboard_dir: Optional directory for tensorboard logs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if tensorboard_dir:
            self.writer = SummaryWriter(tensorboard_dir)
        else:
            self.writer = None
            
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'visualization.log'),
                logging.StreamHandler()
            ]
        )
    
    def plot_training_history(self, 
                            history: Dict[str, list],
                            save_name: str = 'training_history.png',
                            show_plot: bool = False):
        """
        Plot training history with enhanced visualization
        
        Args:
            history: Dictionary containing training metrics
            save_name: Name of the output file
            show_plot: Whether to display the plot
        """
        try:
            fig = plt.figure(figsize=(15, 10))
            
            # Create grid for subplots
            gs = fig.add_gridspec(2, 2, hspace=0.3)
            
            # Accuracy plot
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(history['train_acc'], label='Train', color='#2ecc71')
            ax1.plot(history['val_acc'], label='Validation', color='#e74c3c')
            ax1.set_title('Model Accuracy', pad=20)
            ax1.set_ylabel('Accuracy (%)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Loss plot
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(history['train_loss'], label='Train', color='#2ecc71')
            ax2.plot(history['val_loss'], label='Validation', color='#e74c3c')
            ax2.set_title('Model Loss', pad=20)
            ax2.set_ylabel('Loss')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Learning rate plot
            ax3 = fig.add_subplot(gs[1, :])
            ax3.plot(history['lr'], color='#3498db')
            ax3.set_title('Learning Rate Schedule', pad=20)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.grid(True, alpha=0.3)
            
            # Save plot
            plt.savefig(self.output_dir / save_name, bbox_inches='tight', dpi=300)
            if show_plot:
                plt.show()
            plt.close()
            
            # Log to tensorboard if available
            if self.writer:
                for metric in ['acc', 'loss']:
                    self.writer.add_scalars(
                        f'Metrics/{metric}',
                        {
                            'train': history[f'train_{metric}'][-1],
                            'val': history[f'val_{metric}'][-1]
                        },
                        len(history[f'train_{metric}']) - 1
                    )
            
            logging.info(f"Training history plot saved to {save_name}")
            
        except Exception as e:
            logging.error(f"Error plotting training history: {str(e)}")
            raise
    
    def plot_confusion_matrix(self,
                            cm: np.ndarray,
                            class_names: list,
                            save_name: str = 'confusion_matrix.png',
                            show_plot: bool = False):
        """
        Plot enhanced confusion matrix with additional metrics
        
        Args:
            cm: Confusion matrix array
            class_names: List of class names
            save_name: Name of the output file
            show_plot: Whether to display the plot
        """
        try:
            plt.figure(figsize=(10, 8))
            
            # Calculate metrics
            total = np.sum(cm)
            accuracy = np.trace(cm) / total
            misclass = 1 - accuracy
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names,
                       yticklabels=class_names)
            
            plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2%} | Error: {misclass:.2%}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save plot
            plt.savefig(self.output_dir / save_name, bbox_inches='tight', dpi=300)
            if show_plot:
                plt.show()
            plt.close()
            
            # Log to tensorboard if available
            if self.writer:
                self.writer.add_figure('Confusion Matrix', plt.gcf())
            
            logging.info(f"Confusion matrix plot saved to {save_name}")
            
        except Exception as e:
            logging.error(f"Error plotting confusion matrix: {str(e)}")
            raise
    
    def plot_roc_curve(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      save_name: str = 'roc_curve.png',
                      show_plot: bool = False):
        """
        Plot ROC curve with interactive features
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            save_name: Name of the output file
            show_plot: Whether to display the plot
        """
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            
            # Create interactive plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f'ROC curve (AUC = {roc_auc:.2f})',
                mode='lines',
                line=dict(color='#2ecc71', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name='Random',
                mode='lines',
                line=dict(color='#e74c3c', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title='Receiver Operating Characteristic (ROC) Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                showlegend=True,
                width=800,
                height=600
            )
            
            # Save plot
            fig.write_html(self.output_dir / f"{save_name}.html")
            fig.write_image(self.output_dir / save_name)
            
            if show_plot:
                fig.show()
            
            logging.info(f"ROC curve plot saved to {save_name}")
            
        except Exception as e:
            logging.error(f"Error plotting ROC curve: {str(e)}")
            raise
    
    def plot_feature_importance(self,
                              model: torch.nn.Module,
                              sample_input: torch.Tensor,
                              save_name: str = 'feature_importance.png',
                              show_plot: bool = False):
        """
        Plot feature importance using SHAP values
        
        Args:
            model: Trained PyTorch model
            sample_input: Sample input tensor
            save_name: Name of the output file
            show_plot: Whether to display the plot
        """
        try:
            # Convert model to eval mode
            model.eval()
            
            # Create SHAP explainer
            background = torch.zeros_like(sample_input)
            explainer = shap.DeepExplainer(model, background)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(sample_input)
            
            # Plot SHAP values
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, sample_input.cpu().numpy(),
                            show=False)
            
            # Save plot
            plt.savefig(self.output_dir / save_name, bbox_inches='tight', dpi=300)
            if show_plot:
                plt.show()
            plt.close()
            
            logging.info(f"Feature importance plot saved to {save_name}")
            
        except Exception as e:
            logging.error(f"Error plotting feature importance: {str(e)}")
            raise
    
    def __del__(self):
        """Cleanup resources"""
        if self.writer:
            self.writer.close()

def main():
    """Example usage of the EnhancedVisualizer"""
    # Create dummy data
    history = {
        'train_acc': [0.8, 0.85, 0.9],
        'val_acc': [0.75, 0.8, 0.85],
        'train_loss': [0.5, 0.3, 0.2],
        'val_loss': [0.6, 0.4, 0.3],
        'lr': [0.001, 0.0008, 0.0005]
    }
    
    cm = np.array([[424, 76], [34, 466]])
    class_names = ['Non-Glaucoma', 'Glaucoma']
    
    # Initialize visualizer
    visualizer = EnhancedVisualizer('visualization_output')
    
    # Generate plots
    visualizer.plot_training_history(history)
    visualizer.plot_confusion_matrix(cm, class_names)
    
    logging.info("Visualization examples completed successfully")

if __name__ == "__main__":
    main()