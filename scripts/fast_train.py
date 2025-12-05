#!/usr/bin/env python3
"""
ULTRA-FAST Training - Maximum Speed.
Only saves final visualizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse
from datetime import datetime
from tqdm import tqdm
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.config import Config
from src.data import StockDataLoader


class FastModel(nn.Module):
    """Minimal model for speed."""
    
    def __init__(self, n_stocks, lookback, d_model=32, max_lag=12):
        super().__init__()
        self.n_stocks = n_stocks
        self.lookback = lookback
        self.max_lag = max_lag
        self.min_lag = 2
        self.n_lags = max_lag - self.min_lag + 1
        
        # Simple input projection
        self.input_proj = nn.Linear(2, d_model)
        
        # Per-stock embeddings
        self.stock_emb = nn.Embedding(n_stocks, d_model)
        
        # Single attention layer
        self.attn = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
        # Causal gates (learnable per stock)
        self.gates = nn.Parameter(torch.zeros(n_stocks))
        
        # Lag prediction
        self.lag_net = nn.Linear(d_model, 1)
        
        # Output
        self.out = nn.Linear(d_model, 1)
        
    def forward(self, X_ret, X_vol, target_idx=None):
        B, T, N = X_ret.shape
        device = X_ret.device
        
        # Stack inputs
        x = torch.stack([X_ret, X_vol], dim=-1)  # (B, T, N, 2)
        x = self.input_proj(x)  # (B, T, N, D)
        
        # Add stock embeddings
        stock_ids = torch.arange(N, device=device)
        x = x + self.stock_emb(stock_ids)
        
        # Flatten for attention: (B, T*N, D)
        x_flat = x.view(B, T * N, -1)
        
        # Self-attention
        attn_out, attn_weights = self.attn(x_flat, x_flat, x_flat)
        x_flat = self.norm(x_flat + attn_out)
        
        # Reshape back
        x = x_flat.view(B, T, N, -1)
        
        # Pool over time (last timestep)
        x = x[:, -1, :, :]  # (B, N, D)
        
        # Gates (sigmoid for 0-1)
        gates = torch.sigmoid(self.gates)  # (N,)
        
        # Lag prediction per stock
        lags = torch.sigmoid(self.lag_net(x).squeeze(-1)) * self.max_lag  # (B, N)
        
        # Prediction for target
        if target_idx is not None:
            # Weighted sum of other stocks: gates (N,) -> (1, N, 1)
            weights = gates.view(1, -1, 1) * x  # (B, N, D)
            pred_input = weights.mean(dim=1)  # (B, D)
            pred = self.out(pred_input)
        else:
            pred = self.out(x).squeeze(-1)
        
        return {
            'predictions': pred,
            'gates': gates,
            'lags': lags.mean(0),  # Average over batch
            'attn_weights': attn_weights
        }


def load_data(config, tickers=None):
    """Load data once."""
    loader = StockDataLoader(config.DATA_PATH, config)
    loader.load_data(tickers=tickers)
    loader.compute_realized_volatility()
    
    n = len(loader.returns)
    train_end = int(n * config.TRAIN_SPLIT)
    loader.normalize_data(train_end_idx=train_end)
    
    X_ret, X_vol, y = loader.create_sequences()
    train, val, test = loader.split_data(X_ret, X_vol, y)
    
    return {
        'train': train, 'val': val, 'test': test,
        'stock_names': loader.stock_names,
        'n_stocks': len(loader.stock_names)
    }


def train_stock(idx, name, data, config, epochs=10, batch_size=512):
    """Train one stock - FAST."""
    device = config.DEVICE
    
    # Create tensors
    X_ret, X_vol, y = data['train']
    train_ds = TensorDataset(
        torch.FloatTensor(X_ret),
        torch.FloatTensor(X_vol),
        torch.FloatTensor(y[:, idx])
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    
    X_ret_v, X_vol_v, y_v = data['val']
    val_ds = TensorDataset(
        torch.FloatTensor(X_ret_v),
        torch.FloatTensor(X_vol_v),
        torch.FloatTensor(y_v[:, idx])
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    # Model
    model = FastModel(data['n_stocks'], config.LOOKBACK_WINDOW).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    
    best_loss = float('inf')
    patience = 0
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_r, X_v, y_batch in train_loader:
            X_r, X_v, y_batch = X_r.to(device), X_v.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            out = model(X_r, X_v, idx)
            loss = F.mse_loss(out['predictions'].squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        preds, targets = [], []
        with torch.no_grad():
            for X_r, X_v, y_batch in val_loader:
                X_r, X_v, y_batch = X_r.to(device), X_v.to(device), y_batch.to(device)
                out = model(X_r, X_v, idx)
                val_loss += F.mse_loss(out['predictions'].squeeze(), y_batch).item()
                preds.append(out['predictions'].cpu().numpy())
                targets.append(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        preds = np.concatenate(preds).flatten()
        targets = np.concatenate(targets).flatten()
        
        # R² score
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - targets.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(r2)
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 2:  # Very aggressive
                break
    
    # Load best
    model.load_state_dict(best_state)
    
    # Extract final causal info
    model.eval()
    with torch.no_grad():
        X_r, X_v, _ = data['val']
        X_r = torch.FloatTensor(X_r[:256]).to(device)
        X_v = torch.FloatTensor(X_v[:256]).to(device)
        out = model(X_r, X_v, idx)
        gates = out['gates'].cpu().numpy()
        lags = out['lags'].cpu().numpy()
    
    return {
        'name': name,
        'r2': max(history['val_r2']),
        'gates': gates,
        'lags': lags,
        'history': history,
        'epochs_trained': len(history['train_loss'])
    }


def create_final_visualization(all_results, stock_names, save_path='plots/final_results.png'):
    """Create single comprehensive visualization."""
    os.makedirs('plots', exist_ok=True)
    
    n_stocks = len(all_results)
    
    fig = plt.figure(figsize=(20, 16), facecolor='#0a0a0a')
    
    # 1. R² scores bar chart
    ax1 = fig.add_subplot(2, 2, 1, facecolor='#1a1a2e')
    r2_scores = [r['r2'] for r in all_results]
    names = [r['name'] for r in all_results]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(r2_scores)))
    bars = ax1.barh(names, r2_scores, color=colors)
    ax1.set_xlabel('R² Score', color='white', fontsize=12)
    ax1.set_title('Model Performance by Stock', color='white', fontsize=14, fontweight='bold')
    ax1.tick_params(colors='white')
    ax1.set_xlim([0, max(r2_scores) * 1.1])
    for i, (bar, score) in enumerate(zip(bars, r2_scores)):
        ax1.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', color='white', fontsize=9)
    
    # 2. Causal strength heatmap (average gates across all trained models)
    ax2 = fig.add_subplot(2, 2, 2, facecolor='#1a1a2e')
    gate_matrix = np.zeros((n_stocks, n_stocks))
    for i, r in enumerate(all_results):
        gate_matrix[i, :] = r['gates']
    
    im = ax2.imshow(gate_matrix, cmap='plasma', aspect='auto')
    ax2.set_xticks(range(n_stocks))
    ax2.set_yticks(range(n_stocks))
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8, color='white')
    ax2.set_yticklabels(names, fontsize=8, color='white')
    ax2.set_title('Causal Influence Strength (Gate Values)', color='white', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Source Stock', color='white')
    ax2.set_ylabel('Target Stock', color='white')
    plt.colorbar(im, ax=ax2, label='Gate Value')
    
    # 3. Lag distribution
    ax3 = fig.add_subplot(2, 2, 3, facecolor='#1a1a2e')
    all_lags = []
    for r in all_results:
        all_lags.extend(r['lags'].tolist())
    ax3.hist(all_lags, bins=20, color='#3498db', edgecolor='white', alpha=0.7)
    ax3.axvline(np.mean(all_lags), color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_lags):.1f}')
    ax3.set_xlabel('Lag (5-min intervals)', color='white', fontsize=12)
    ax3.set_ylabel('Frequency', color='white', fontsize=12)
    ax3.set_title('Distribution of Learned Lags', color='white', fontsize=14, fontweight='bold')
    ax3.tick_params(colors='white')
    ax3.legend(facecolor='#2d2d44', labelcolor='white')
    
    # 4. Training curves (overlay all stocks)
    ax4 = fig.add_subplot(2, 2, 4, facecolor='#1a1a2e')
    for i, r in enumerate(all_results):
        alpha = 0.5 + 0.5 * (i / len(all_results))
        ax4.plot(r['history']['val_loss'], alpha=alpha, linewidth=1)
    ax4.set_xlabel('Epoch', color='white', fontsize=12)
    ax4.set_ylabel('Validation Loss', color='white', fontsize=12)
    ax4.set_title('Training Convergence (All Stocks)', color='white', fontsize=14, fontweight='bold')
    ax4.tick_params(colors='white')
    ax4.set_yscale('log')
    
    plt.suptitle('🎯 Causal Volatility Transmission - Final Results', 
                 fontsize=18, fontweight='bold', color='white', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    print(f"✓ Final visualization saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()
    
    config = Config()
    
    tickers = [t.strip().upper() for t in args.tickers.split(',')] if args.tickers else None
    
    print("\n" + "="*60)
    print("⚡ ULTRA-FAST TRAINING")
    print("="*60)
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Early stopping: 2 epochs")
    print("="*60 + "\n")
    
    # Load data ONCE
    data = load_data(config, tickers)
    print(f"Loaded {data['n_stocks']} stocks\n")
    
    # Train all stocks
    results = []
    start = time.time()
    
    for idx, name in enumerate(tqdm(data['stock_names'], desc="Training")):
        r = train_stock(idx, name, data, config, args.epochs, args.batch_size)
        results.append(r)
        tqdm.write(f"  {name}: R²={r['r2']:.4f} ({r['epochs_trained']} epochs)")
    
    elapsed = time.time() - start
    
    # Summary
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Avg R²: {np.mean([r['r2'] for r in results]):.4f}")
    print(f"{'='*60}\n")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    # Causal relationships
    rows = []
    for r in results:
        for j, source in enumerate(data['stock_names']):
            if r['gates'][j] > 0.1:
                rows.append({
                    'source': source,
                    'target': r['name'],
                    'strength': float(r['gates'][j]),
                    'lag_intervals': float(r['lags'][j]),
                    'lag_minutes': float(r['lags'][j]) * 5
                })
    
    df = pd.DataFrame(rows).sort_values('strength', ascending=False)
    df.to_csv('results/causal_relationships.csv', index=False)
    print(f"✓ Saved {len(df)} causal relationships to results/causal_relationships.csv")
    
    # Create FINAL visualization
    create_final_visualization(results, data['stock_names'])
    
    # Save training summary
    summary = pd.DataFrame([{'stock': r['name'], 'r2': r['r2'], 'epochs': r['epochs_trained']} for r in results])
    summary.to_csv(f'results/training_summary_{datetime.now():%Y%m%d_%H%M}.csv', index=False)


if __name__ == '__main__':
    main()
