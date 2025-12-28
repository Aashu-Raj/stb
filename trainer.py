"""
Model Trainer Module
Handles training, validation, and evaluation
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import json


class RealEstateTrainer:
    """
    Trainer for real estate valuation models
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device='cuda',
        model_name='model',
        save_dir='models'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'train_rmse': [],
            'val_rmse': [],
            'train_mape': [],
            'val_mape': []
        }
        
        self.best_val_loss = float('inf')
        
    def calculate_metrics(self, predictions, targets):
        """Calculate regression metrics"""
        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        
        # MAE
        mae = np.mean(np.abs(predictions - targets))
        
        # RMSE
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        
        # MAPE
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        # R2 Score
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for batch in pbar:
            # Move data to device
            if 'image' in batch:
                images = batch['image'].to(self.device)
                tabular = batch['tabular'].to(self.device)
                targets = batch['target'].to(self.device).unsqueeze(1)
                
                # Forward pass
                predictions = self.model(images, tabular)
            else:
                # Tabular only
                tabular = batch['tabular'].to(self.device)
                targets = batch['target'].to(self.device).unsqueeze(1)
                
                predictions = self.model(tabular)
            
            # Calculate loss
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            all_predictions.append(predictions)
            all_targets.append(targets)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        metrics = self.calculate_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                if 'image' in batch:
                    images = batch['image'].to(self.device)
                    tabular = batch['tabular'].to(self.device)
                    targets = batch['target'].to(self.device).unsqueeze(1)
                    
                    predictions = self.model(images, tabular)
                else:
                    tabular = batch['tabular'].to(self.device)
                    targets = batch['target'].to(self.device).unsqueeze(1)
                    
                    predictions = self.model(tabular)
                
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        avg_loss = total_loss / len(self.val_loader)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        metrics = self.calculate_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def train(self, epochs, scheduler=None, early_stopping_patience=10):
        """
        Full training loop
        
        Args:
            epochs: Number of epochs
            scheduler: Learning rate scheduler (optional)
            early_stopping_patience: Patience for early stopping
        """
        print(f"\nStarting training for {epochs} epochs...")
        print("=" * 70)
        
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 70)
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['train_rmse'].append(train_metrics['rmse'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            self.history['train_mape'].append(train_metrics['mape'])
            self.history['val_mape'].append(val_metrics['mape'])
            
            # Print metrics
            print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train MAE: ${train_metrics['mae']:,.2f} | Val MAE: ${val_metrics['mae']:,.2f}")
            print(f"Train RMSE: ${train_metrics['rmse']:,.2f} | Val RMSE: ${val_metrics['rmse']:,.2f}")
            print(f"Train MAPE: {train_metrics['mape']:.2f}% | Val MAPE: {val_metrics['mape']:.2f}%")
            print(f"Train R²: {train_metrics['r2']:.4f} | Val R²: {val_metrics['r2']:.4f}")
            
            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step(val_loss)
                print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best')
                print(f"✓ Best model saved (Val Loss: {val_loss:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        self.save_checkpoint('final')
        
        # Save training history
        self.save_history()
        
        print("\n" + "=" * 70)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return self.history
    
    def save_checkpoint(self, name='checkpoint'):
        """Save model checkpoint"""
        checkpoint_path = self.save_dir / f"{self.model_name}_{name}.pth"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }, checkpoint_path)
    
    def load_checkpoint(self, name='best'):
        """Load model checkpoint"""
        checkpoint_path = self.save_dir / f"{self.model_name}_{name}.pth"
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
    
    def save_history(self):
        """Save training history"""
        history_path = self.save_dir / f"{self.model_name}_history.json"
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_history(self, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss over Epochs')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE
        axes[0, 1].plot(self.history['train_mae'], label='Train')
        axes[0, 1].plot(self.history['val_mae'], label='Validation')
        axes[0, 1].set_title('MAE over Epochs')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE ($)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # RMSE
        axes[1, 0].plot(self.history['train_rmse'], label='Train')
        axes[1, 0].plot(self.history['val_rmse'], label='Validation')
        axes[1, 0].set_title('RMSE over Epochs')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE ($)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # MAPE
        axes[1, 1].plot(self.history['train_mape'], label='Train')
        axes[1, 1].plot(self.history['val_mape'], label='Validation')
        axes[1, 1].set_title('MAPE over Epochs')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def predict(model, data_loader, device='cuda'):
    """
    Make predictions on data
    
    Args:
        model: Trained model
        data_loader: DataLoader
        device: Device to use
        
    Returns:
        predictions, ids
    """
    model.eval()
    all_predictions = []
    all_ids = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Predicting'):
            if 'image' in batch:
                images = batch['image'].to(device)
                tabular = batch['tabular'].to(device)
                predictions = model(images, tabular)
            else:
                tabular = batch['tabular'].to(device)
                predictions = model(tabular)
            
            all_predictions.append(predictions.cpu())
            all_ids.extend(batch['id'].tolist())
    
    all_predictions = torch.cat(all_predictions).squeeze().numpy()
    
    return all_predictions, all_ids