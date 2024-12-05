import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime
import random
from losses import FocalLoss
from Model import get_model
from DataLoader import get_data_loaders
from torch.optim.lr_scheduler import OneCycleLR
import os

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.criterion = FocalLoss(alpha=0.25, gamma=2)
        self.device = next(model.parameters()).device
        
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.scaler = GradScaler()
        
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        self.early_stopping = EarlyStopping(
            patience=config['early_stopping_patience'],
            verbose=True
        )
        
        self.best_val_acc = 0.0
        self._setup_logging()
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

    def _get_optimizer(self):
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

    def _get_scheduler(self):
        return OneCycleLR(
            self.optimizer,
            max_lr=self.config['max_lr'],
            epochs=self.config['num_epochs'],
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            final_div_factor=1e3
        )

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config['log_dir'] / 'training.log'),
                logging.StreamHandler()
            ]
        )

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(device_type=self.device.type):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.scheduler.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss/(batch_idx+1),
                'acc': 100.*correct/total,
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        return running_loss/len(self.train_loader), 100.*correct/total

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in self.val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            with autocast(device_type=self.device.type):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return running_loss/len(self.val_loader), 100.*correct/total

    def train(self):
        for epoch in range(self.config['num_epochs']):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            # Logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            logging.info(f'Epoch {epoch+1}: '
                         f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                         f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                }, self.config['model_dir'] / 'best_model.pth')
            
            # Early stopping
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        self.writer.close()
        return {
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'epochs_trained': epoch + 1
        }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_model_state(model, trainer, config, save_dir='models'):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = save_dir / f"model_state.pth"
    
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'best_val_acc': trainer.best_val_acc,
        'config': config
    }
    
    torch.save(state, save_path)
    logging.info(f"Model state saved to {save_path}")
    return save_path

def save_metadata(config, results, save_dir='models'):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = save_dir / f"metadata.json"
    
    metadata = {
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
        'best_val_acc': results['best_val_acc'],
        'num_epochs_trained': results['epochs_trained'],
        'final_train_loss': results['history']['train_loss'][-1],
        'final_train_acc': results['history']['train_acc'][-1],
        'final_val_loss': results['history']['val_loss'][-1],
        'final_val_acc': results['history']['val_acc'][-1],
        'final_learning_rate': results['history']['lr'][-1],
        'early_stopping_triggered': results['epochs_trained'] < config['num_epochs']
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logging.info(f"Metadata saved to {metadata_path}")
    return metadata_path

def main():
    # Configuration
    config = {
        'data_dir': Path('./data/raw'),
        'log_dir': Path('logs/run_001'),
        'model_dir': Path('models'),
        'num_epochs': 50,
        'batch_size': 64,
        'learning_rate': 1e-6,
        'max_lr': 1e-5,
        'weight_decay': 1e-5,
        'early_stopping_patience': 10,
        'seed': 42
    }

    # Set up directories
    config['log_dir'].mkdir(parents=True, exist_ok=True)
    config['model_dir'].mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    set_seed(config['seed'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(config['data_dir'], config['batch_size'])

    # Initialize model
    model = get_model(device)

    # Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, config)

    # Train the model
    results = trainer.train()

    # Save final model state
    model_path = save_model_state(model, trainer, config)

    # Save training history
    history_path = config['model_dir'] / 'training_history.npy'
    np.save(history_path, results['history'])
    logging.info(f"Training history saved to {history_path}")

    # Save metadata
    metadata_path = save_metadata(config, results)

    logging.info(f"Training completed after {results['epochs_trained']} epochs.")
    logging.info(f"Best validation accuracy: {results['best_val_acc']:.2f}%")
    logging.info(f"Model saved to: {model_path}")
    logging.info(f"Training history saved to: {history_path}")
    logging.info(f"Metadata saved to: {metadata_path}")

    if results['epochs_trained'] < config['num_epochs']:
        logging.info("Note: Training stopped early due to early stopping criterion.")

if __name__ == '__main__':
    main()