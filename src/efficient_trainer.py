"""
Efficient Trainer with Mixed Precision, Training Visualization, and GPU Optimization.

Key features:
- Mixed precision (AMP) for 2-3x speedup on GPU
- Gradient accumulation for effective larger batch sizes
- Live training visualization
- Learning rate scheduling with warmup
- Comprehensive metrics tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict
from datetime import datetime
import time
import os
import json
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from src.config import Config
from src.data import StockDataLoader, VolatilityDataset
from src.models.efficient_attention import EfficientCausalModel, EfficientLoss
from src.utils import calculate_metrics


class TrainingVisualizer:
    """Real-time training visualization for deep learning class presentations."""
    
    def __init__(self, stock_name: str, save_dir: str = 'plots/training/', 
                 update_freq: int = 5):
        """
        Initialize visualizer.
        
        Args:
            stock_name: Name of stock being trained
            save_dir: Directory to save plots
            update_freq: How often to update plots (in epochs)
        """
        self.stock_name = stock_name
        self.save_dir = save_dir
        self.update_freq = update_freq
        os.makedirs(save_dir, exist_ok=True)
        
        # History tracking
        self.history = defaultdict(list)
        
        # Setup figure
        plt.style.use('dark_background')
        
    def update(self, epoch: int, train_metrics: Dict, val_metrics: Dict,
               model: nn.Module = None, learning_rate: float = None):
        """
        Update training history and optionally create visualizations.
        
        Args:
            epoch: Current epoch number
            train_metrics: Dictionary of training metrics
            val_metrics: Dictionary of validation metrics
            model: Model for extracting attention weights
            learning_rate: Current learning rate
        """
        # Store metrics
        self.history['epoch'].append(epoch)
        
        for key, value in train_metrics.items():
            self.history[f'train_{key}'].append(value)
        for key, value in val_metrics.items():
            self.history[f'val_{key}'].append(value)
            
        if learning_rate is not None:
            self.history['learning_rate'].append(learning_rate)
        
        # Create visualization periodically
        if epoch % self.update_freq == 0 or epoch == 1:
            self._create_dashboard(epoch, model)
    
    def _create_dashboard(self, epoch: int, model: nn.Module = None):
        """Create comprehensive training dashboard."""
        fig = plt.figure(figsize=(20, 12), facecolor='#0f0f23')
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
        
        epochs = self.history['epoch']
        
        # 1. Loss curves
        ax1 = fig.add_subplot(gs[0, 0], facecolor='#1a1a2e')
        if 'train_mse_loss' in self.history:
            ax1.plot(epochs, self.history['train_mse_loss'], 'o-', 
                    color='#3498db', linewidth=2, markersize=3, label='Train MSE')
        if 'val_mse_loss' in self.history:
            ax1.plot(epochs, self.history['val_mse_loss'], 's-',
                    color='#e74c3c', linewidth=2, markersize=3, label='Val MSE')
        ax1.set_xlabel('Epoch', color='white', fontsize=11)
        ax1.set_ylabel('MSE Loss', color='white', fontsize=11)
        ax1.set_title('📉 Loss Curves', color='white', fontsize=14, fontweight='bold')
        ax1.legend(facecolor='#2d2d44', labelcolor='white')
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.2)
        ax1.set_yscale('log')
        
        # 2. R² Score progression
        ax2 = fig.add_subplot(gs[0, 1], facecolor='#1a1a2e')
        if 'val_r2' in self.history:
            ax2.fill_between(epochs, 0, self.history['val_r2'], 
                           alpha=0.3, color='#2ecc71')
            ax2.plot(epochs, self.history['val_r2'], 'o-',
                    color='#2ecc71', linewidth=2.5, markersize=4)
            # Add best R² annotation
            best_r2 = max(self.history['val_r2'])
            best_epoch = epochs[self.history['val_r2'].index(best_r2)]
            ax2.axhline(y=best_r2, color='#27ae60', linestyle='--', alpha=0.5)
            ax2.annotate(f'Best: {best_r2:.4f}', xy=(best_epoch, best_r2),
                        color='#2ecc71', fontsize=10, fontweight='bold')
        ax2.set_xlabel('Epoch', color='white', fontsize=11)
        ax2.set_ylabel('R² Score', color='white', fontsize=11)
        ax2.set_title('📈 Prediction Quality (R²)', color='white', fontsize=14, fontweight='bold')
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.2)
        ax2.set_ylim([0, max(1, max(self.history.get('val_r2', [0.5])) * 1.1)])
        
        # 3. Learning rate schedule
        ax3 = fig.add_subplot(gs[0, 2], facecolor='#1a1a2e')
        if 'learning_rate' in self.history:
            ax3.plot(epochs, self.history['learning_rate'], 'o-',
                    color='#9b59b6', linewidth=2, markersize=4)
            ax3.fill_between(epochs, 0, self.history['learning_rate'],
                           alpha=0.2, color='#9b59b6')
        ax3.set_xlabel('Epoch', color='white', fontsize=11)
        ax3.set_ylabel('Learning Rate', color='white', fontsize=11)
        ax3.set_title('⚡ Learning Rate Schedule', color='white', fontsize=14, fontweight='bold')
        ax3.tick_params(colors='white')
        ax3.grid(True, alpha=0.2)
        ax3.set_yscale('log')
        
        # 4. Loss components
        ax4 = fig.add_subplot(gs[1, 0], facecolor='#1a1a2e')
        components = ['gate_loss', 'contrastive_loss', 'cluster_loss']
        colors = ['#e74c3c', '#3498db', '#f39c12']
        for comp, color in zip(components, colors):
            key = f'train_{comp}'
            if key in self.history:
                ax4.plot(epochs, self.history[key], 'o-', 
                        color=color, linewidth=2, markersize=3, label=comp.replace('_', ' ').title())
        ax4.set_xlabel('Epoch', color='white', fontsize=11)
        ax4.set_ylabel('Loss Value', color='white', fontsize=11)
        ax4.set_title('🔧 Loss Components', color='white', fontsize=14, fontweight='bold')
        ax4.legend(facecolor='#2d2d44', labelcolor='white', fontsize=9)
        ax4.tick_params(colors='white')
        ax4.grid(True, alpha=0.2)
        
        # 5. RMSE over time
        ax5 = fig.add_subplot(gs[1, 1], facecolor='#1a1a2e')
        if 'val_rmse' in self.history:
            ax5.plot(epochs, self.history['val_rmse'], 's-',
                    color='#1abc9c', linewidth=2.5, markersize=4)
            ax5.fill_between(epochs, max(self.history['val_rmse']),
                           self.history['val_rmse'], alpha=0.3, color='#1abc9c')
        if 'train_rmse' in self.history:
            ax5.plot(epochs, self.history['train_rmse'], 'o-',
                    color='#3498db', linewidth=2, markersize=3, alpha=0.7)
        ax5.set_xlabel('Epoch', color='white', fontsize=11)
        ax5.set_ylabel('RMSE', color='white', fontsize=11)
        ax5.set_title('📊 Root Mean Square Error', color='white', fontsize=14, fontweight='bold')
        ax5.tick_params(colors='white')
        ax5.grid(True, alpha=0.2)
        
        # 6. Statistics summary
        ax6 = fig.add_subplot(gs[1, 2], facecolor='#1a1a2e')
        ax6.axis('off')
        
        stats_text = f"""
╔════════════════════════════════════╗
║   TRAINING STATISTICS              ║
╠════════════════════════════════════╣
║ Stock: {self.stock_name:27s} ║
║ Epoch: {epoch:27d} ║
╠════════════════════════════════════╣
"""
        if self.history['val_mse_loss']:
            best_loss = min(self.history['val_mse_loss'])
            curr_loss = self.history['val_mse_loss'][-1]
            stats_text += f"║ Best Val Loss: {best_loss:19.6f} ║\n"
            stats_text += f"║ Current Val Loss: {curr_loss:16.6f} ║\n"
        
        if self.history.get('val_r2'):
            best_r2 = max(self.history['val_r2'])
            curr_r2 = self.history['val_r2'][-1]
            stats_text += f"║ Best R²: {best_r2:25.4f} ║\n"
            stats_text += f"║ Current R²: {curr_r2:22.4f} ║\n"
        
        if self.history.get('learning_rate'):
            curr_lr = self.history['learning_rate'][-1]
            stats_text += f"║ Learning Rate: {curr_lr:19.2e} ║\n"
        
        stats_text += "╚════════════════════════════════════╝"
        
        ax6.text(0.5, 0.5, stats_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='center', horizontalalignment='center',
                family='monospace', color='#2ecc71',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f0f23',
                         edgecolor='#2ecc71', linewidth=2))
        
        # Main title
        fig.suptitle(f'🎯 Deep Learning Training Dashboard: {self.stock_name}',
                    fontsize=18, fontweight='bold', color='white', y=0.98)
        
        plt.tight_layout()
        
        # Save
        save_path = f"{self.save_dir}{self.stock_name}_training_epoch_{epoch:03d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f0f23')
        plt.close()
        
    def save_final(self) -> str:
        """Create and save final training summary."""
        self._create_dashboard(len(self.history['epoch']), None)
        
        # Also save history as JSON
        history_path = f"{self.save_dir}{self.stock_name}_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)
        
        return history_path


class CosineWarmupScheduler:
    """Learning rate scheduler with linear warmup and cosine decay."""
    
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr_scale = 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = max(self.min_lr, base_lr * lr_scale)
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


class EfficientTrainer:
    """
    Highly efficient trainer with:
    - Mixed precision (AMP)
    - Gradient accumulation
    - Learning rate scheduling
    - Live training visualization
    """
    
    def __init__(self, config: Config, target_stock_idx: int, stock_name: str,
                 use_amp: bool = True, gradient_accumulation_steps: int = 1,
                 visualize: bool = True, viz_update_freq: int = 5):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
            target_stock_idx: Index of target stock
            stock_name: Name of target stock
            use_amp: Use automatic mixed precision
            gradient_accumulation_steps: Number of steps to accumulate gradients
            visualize: Enable training visualization
            viz_update_freq: How often to update visualizations
        """
        self.config = config
        self.target_stock_idx = target_stock_idx
        self.stock_name = stock_name
        self.device = config.DEVICE
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        
        # Visualization
        self.visualizer = None
        if visualize:
            self.visualizer = TrainingVisualizer(stock_name, update_freq=viz_update_freq)
    
    def setup_model(self, n_stocks: int, n_layers: int = 2, n_heads: int = 4,
                    d_model: int = None):
        """
        Setup model with optimized architecture.
        
        Args:
            n_stocks: Number of stocks
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            d_model: Model dimension (uses config if not specified)
        """
        d_model = d_model or self.config.D_MODEL
        
        self.model = EfficientCausalModel(
            n_stocks=n_stocks,
            lookback=self.config.LOOKBACK_WINDOW,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_lag=self.config.MAX_LAG,
            latent_dim=min(32, d_model // 2),
            n_clusters=min(8, n_stocks // 4 + 1),
            dropout=self.config.DROPOUT
        ).to(self.device)
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=5,
            total_epochs=self.config.NUM_EPOCHS
        )
        
        # Loss function
        self.criterion = EfficientLoss(
            lambda_gate=self.config.LAMBDA_GATE,
            lambda_contrastive=0.01,
            lambda_cluster=0.001
        )
        
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model initialized: {n_params:,} parameters")
        if self.use_amp:
            print("Mixed precision training ENABLED (2-3x speedup)")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """Train for one epoch with AMP and gradient accumulation."""
        self.model.train()
        epoch_metrics = defaultdict(float)
        n_batches = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for batch_idx, (X_returns, X_volatility, y) in enumerate(pbar):
            X_returns = X_returns.to(self.device, non_blocking=True)
            X_volatility = X_volatility.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                outputs = self.model(X_returns, X_volatility, self.target_stock_idx)
                loss_dict = self.criterion(
                    outputs['predictions'], y,
                    outputs['gates'], outputs['latent'],
                    outputs['cluster_probs']
                )
                loss = loss_dict['total_loss'] / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Track metrics
            for key, value in loss_dict.items():
                if key != 'total_loss':
                    epoch_metrics[key] += value
            n_batches += 1
            
            pbar.set_postfix({'loss': f"{loss_dict['mse_loss']:.4f}"})
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
        
        return dict(epoch_metrics)
    
    def validate(self, val_loader: DataLoader) -> Tuple[Dict, Dict]:
        """Validate the model."""
        self.model.eval()
        val_metrics = defaultdict(float)
        all_predictions = []
        all_targets = []
        n_batches = 0
        
        with torch.no_grad():
            for X_returns, X_volatility, y in val_loader:
                X_returns = X_returns.to(self.device, non_blocking=True)
                X_volatility = X_volatility.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                
                with autocast(enabled=self.use_amp):
                    outputs = self.model(X_returns, X_volatility, self.target_stock_idx)
                    loss_dict = self.criterion(
                        outputs['predictions'], y,
                        outputs['gates'], outputs['latent'],
                        outputs['cluster_probs']
                    )
                
                for key, value in loss_dict.items():
                    if key != 'total_loss':
                        val_metrics[key] += value
                n_batches += 1
                
                all_predictions.append(outputs['predictions'].float().cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= n_batches
        
        # Calculate regression metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        regression_metrics = calculate_metrics(all_predictions, all_targets)
        
        return dict(val_metrics), regression_metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              early_stopping_patience: int = None) -> Dict:
        """
        Full training loop with visualization.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            early_stopping_patience: Patience for early stopping (None = from config)
            
        Returns:
            Training history dictionary
        """
        if early_stopping_patience is None:
            early_stopping_patience = self.config.EARLY_STOPPING_PATIENCE
        
        print(f"\n{'='*70}")
        print(f"TRAINING: {self.stock_name} (idx {self.target_stock_idx})")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"AMP: {'Enabled' if self.use_amp else 'Disabled'}")
        print(f"Epochs: {self.config.NUM_EPOCHS}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            epoch_start = time.time()
            
            # Update learning rate
            self.scheduler.step(epoch - 1)
            current_lr = self.scheduler.get_lr()
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_loss_dict, val_regression = self.validate(val_loader)
            
            # Combine metrics
            val_metrics = {**val_loss_dict, **val_regression}
            
            epoch_time = time.time() - epoch_start
            
            # Print progress
            print(f"Epoch {epoch:3d}/{self.config.NUM_EPOCHS} | "
                  f"Train MSE: {train_metrics['mse_loss']:.5f} | "
                  f"Val MSE: {val_metrics['mse_loss']:.5f} | "
                  f"Val R²: {val_metrics['r2']:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Add RMSE to train metrics for visualization
            train_metrics['rmse'] = np.sqrt(train_metrics['mse_loss'])
            val_metrics['rmse'] = val_metrics.get('rmse', np.sqrt(val_metrics['mse_loss']))
            
            # Update visualization
            if self.visualizer:
                self.visualizer.update(epoch, train_metrics, val_metrics, 
                                       self.model, current_lr)
            
            # Track history
            self.train_losses.append(train_metrics)
            self.val_losses.append(val_metrics)
            
            # Early stopping check
            if val_metrics['mse_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['mse_loss']
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.patience_counter = 0
                self.save_checkpoint('best')
                print(f"  ✓ New best model saved! (Val MSE: {self.best_val_loss:.6f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= early_stopping_patience:
                    print(f"\n⚠️  Early stopping at epoch {epoch} (patience={early_stopping_patience})")
                    break
        
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best Val MSE: {self.best_val_loss:.6f}")
        
        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        # Save final visualization
        if self.visualizer:
            history_path = self.visualizer.save_final()
            print(f"Training history saved to: {history_path}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_time': total_time
        }
    
    def save_checkpoint(self, name: str = 'checkpoint'):
        """Save model checkpoint."""
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'target_stock_idx': self.target_stock_idx,
            'stock_name': self.stock_name,
            'config': {
                'D_MODEL': self.config.D_MODEL,
                'LOOKBACK_WINDOW': self.config.LOOKBACK_WINDOW,
                'MAX_LAG': self.config.MAX_LAG
            }
        }
        path = f'checkpoints/{self.stock_name}_{name}.pt'
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, name: str = 'checkpoint'):
        """Load model checkpoint."""
        path = f'checkpoints/{self.stock_name}_{name}.pt'
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded checkpoint from {path}")
            return True
        return False
    
    def get_causal_graph(self, val_loader: DataLoader) -> Dict:
        """Extract causal relationships from trained model."""
        self.model.eval()
        
        # Get a batch of data
        X_returns, X_volatility, _ = next(iter(val_loader))
        X_returns = X_returns.to(self.device)
        X_volatility = X_volatility.to(self.device)
        
        # Extract causal graph
        return self.model.get_causal_graph(X_returns, X_volatility)


def run_efficient_training(num_stocks: int = None, epochs: int = 30,
                          batch_size: int = 64, visualize: bool = True) -> Dict:
    """
    Quick function to run efficient training on a single stock.
    
    Args:
        num_stocks: Number of stocks to load
        epochs: Number of epochs
        batch_size: Batch size
        visualize: Enable visualization
        
    Returns:
        Training results dictionary
    """
    config = Config()
    config.NUM_EPOCHS = epochs
    config.BATCH_SIZE = batch_size
    
    print("Loading data...")
    data_loader = StockDataLoader(config.DATA_PATH, config)
    data_loader.load_data(max_stocks=num_stocks)
    data_loader.compute_realized_volatility()
    
    n_timesteps = len(data_loader.returns)
    train_end_idx = int(n_timesteps * config.TRAIN_SPLIT)
    data_loader.normalize_data(train_end_idx=train_end_idx)
    
    X_returns, X_volatility, y = data_loader.create_sequences()
    train_data, val_data, test_data = data_loader.split_data(X_returns, X_volatility, y)
    
    # Target first stock
    target_idx = 0
    stock_name = data_loader.stock_names[target_idx]
    
    # Create datasets
    train_dataset = VolatilityDataset(*train_data, target_stock_idx=target_idx)
    val_dataset = VolatilityDataset(*val_data, target_stock_idx=target_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=0, pin_memory=True)
    
    # Train
    trainer = EfficientTrainer(config, target_idx, stock_name, visualize=visualize)
    trainer.setup_model(n_stocks=len(data_loader.stock_names))
    result = trainer.train(train_loader, val_loader)
    
    return result


if __name__ == '__main__':
    # Quick test
    result = run_efficient_training(num_stocks=30, epochs=20, visualize=True)
    print(f"\nFinal Result: Best Val MSE = {result['best_val_loss']:.6f}")

