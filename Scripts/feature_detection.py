import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import torch
from torchvision.utils import make_grid
import logging
from typing import Dict, List, Optional
import seaborn as sns

class GlaucomaFeatureVisualizer:
    """Visualize glaucoma features and track detection performance across epochs"""
    
    def __init__(self, output_dir: str = 'visualization_output'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_history = {
            'cdr_values': [],
            'rim_widths': [],
            'hemorrhage_counts': [],
            'rnfl_defect_counts': [],
            'ppa_ratios': [],
            'risk_distributions': []
        }
        self.logger = logging.getLogger(__name__)

    def visualize_detection_results(
        self,
        image: np.ndarray,
        features: Dict,
        epoch: int,
        sample_idx: int
    ) -> None:
        """Visualize detected features on the image"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Display original image
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Draw optic disc
        if 'disc_params' in features:
            circle = plt.Circle(
                features['disc_params']['center'],
                features['disc_params']['radius'],
                color='blue',
                fill=False,
                linewidth=2,
                label='Optic Disc'
            )
            ax.add_patch(circle)
        
        # Draw optic cup
        if 'cup_params' in features:
            ellipse = plt.matplotlib.patches.Ellipse(
                features['cup_params']['center'],
                features['cup_params']['axes'][0] * 2,
                features['cup_params']['axes'][1] * 2,
                angle=features['cup_params']['angle'],
                color='red',
                fill=False,
                linewidth=2,
                label='Optic Cup'
            )
            ax.add_patch(ellipse)
        
        # Draw hemorrhages
        for hem in features.get('hemorrhages', []):
            rect = plt.Rectangle(
                hem['position'],
                hem['size'][0],
                hem['size'][1],
                color='yellow',
                fill=False,
                linewidth=2
            )
            ax.add_patch(rect)
        
        # Draw RNFL defects
        for defect in features.get('rnfl_defects', []):
            rect = plt.Rectangle(
                defect['position'],
                defect['size'][0],
                defect['size'][1],
                color='green',
                fill=False,
                linewidth=2
            )
            ax.add_patch(rect)
        
        # Add text annotations
        cdr = features.get('cdr', {}).get('vertical_cdr', 0)
        risk = features.get('glaucoma_risk', 'unknown')
        
        plt.text(
            0.02, 0.98,
            f'CDR: {cdr:.2f}\nRisk: {risk}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.title(f'Epoch {epoch} - Sample {sample_idx}')
        plt.axis('off')
        plt.legend()
        
        # Save visualization
        save_path = self.output_dir / f'epoch_{epoch}_sample_{sample_idx}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    def update_history(self, epoch_features: List[Dict]) -> None:
        """Update feature history with results from current epoch"""
        cdr_values = []
        rim_widths = []
        hemorrhage_counts = []
        rnfl_counts = []
        ppa_ratios = []
        risks = {'low': 0, 'moderate': 0, 'high': 0}
        
        for features in epoch_features:
            if 'cdr' in features:
                cdr_values.append(features['cdr'].get('vertical_cdr', 0))
            rim_widths.append(features.get('neuroretinal_rim_width', 0))
            hemorrhage_counts.append(len(features.get('hemorrhages', [])))
            rnfl_counts.append(len(features.get('rnfl_defects', [])))
            ppa_ratios.append(features.get('peripapillary_atrophy', {}).get('area_ratio', 0))
            risks[features.get('glaucoma_risk', 'low')] += 1
        
        self.feature_history['cdr_values'].append(np.mean(cdr_values))
        self.feature_history['rim_widths'].append(np.mean(rim_widths))
        self.feature_history['hemorrhage_counts'].append(np.mean(hemorrhage_counts))
        self.feature_history['rnfl_defect_counts'].append(np.mean(rnfl_counts))
        self.feature_history['ppa_ratios'].append(np.mean(ppa_ratios))
        self.feature_history['risk_distributions'].append(risks)

    def plot_feature_trends(self, epochs: List[int]) -> None:
        """Plot trends of various features across epochs"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 20))
        
        # Plot CDR trend
        axes[0, 0].plot(epochs, self.feature_history['cdr_values'])
        axes[0, 0].set_title('Average CDR Across Epochs')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('CDR')
        
        # Plot rim width trend
        axes[0, 1].plot(epochs, self.feature_history['rim_widths'])
        axes[0, 1].set_title('Average Rim Width Across Epochs')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Width')
        
        # Plot hemorrhage counts
        axes[1, 0].plot(epochs, self.feature_history['hemorrhage_counts'])
        axes[1, 0].set_title('Average Hemorrhage Count Across Epochs')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Count')
        
        # Plot RNFL defect counts
        axes[1, 1].plot(epochs, self.feature_history['rnfl_defect_counts'])
        axes[1, 1].set_title('Average RNFL Defect Count Across Epochs')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Count')
        
        # Plot PPA ratios
        axes[2, 0].plot(epochs, self.feature_history['ppa_ratios'])
        axes[2, 0].set_title('Average PPA Ratio Across Epochs')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Ratio')
        
        # Plot risk distribution
        risk_data = np.array([[d['low'], d['moderate'], d['high']] 
                            for d in self.feature_history['risk_distributions']])
        axes[2, 1].stackplot(epochs, risk_data.T, 
                           labels=['Low', 'Moderate', 'High'])
        axes[2, 1].set_title('Risk Distribution Across Epochs')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Number of Cases')
        axes[2, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_trends.png')
        plt.close()

    def save_epoch_summary(self, epoch: int, features: List[Dict]) -> None:
        """Save statistical summary of features for the epoch"""
        summary = {
            'epoch': epoch,
            'cdr_stats': {
                'mean': np.mean([f['cdr']['vertical_cdr'] for f in features if 'cdr' in f]),
                'std': np.std([f['cdr']['vertical_cdr'] for f in features if 'cdr' in f])
            },
            'risk_distribution': {
                'low': sum(1 for f in features if f.get('glaucoma_risk') == 'low'),
                'moderate': sum(1 for f in features if f.get('glaucoma_risk') == 'moderate'),
                'high': sum(1 for f in features if f.get('glaucoma_risk') == 'high')
            }
        }
        
        save_path = self.output_dir / f'epoch_{epoch}_summary.json'
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=4)