#!/usr/bin/env python3
"""
ULTRA-FAST Training - Attention weights = Causal Strength.
No separate gates - attention IS the causal mechanism.
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


class AttentionCausalModel(nn.Module):
    """
    FAST causal model - attention weights = causal strength.
    Optimized for speed.
    """
    
    def __init__(self, n_stocks, lookback, d_model=16, n_heads=1, max_lag=12):
        super().__init__()
        self.n_stocks = n_stocks
        self.lookback = lookback
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_lag = max_lag
        self.min_lag = 2
        
        # Input projection
        self.input_proj = nn.Linear(2, d_model)
        
        # Stock embeddings
        self.stock_emb = nn.Embedding(n_stocks, d_model)
        
        # Position embeddings
        self.pos_emb = nn.Embedding(lookback, d_model)
        
        # Single attention layer (this IS the causal mechanism)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(d_model)
        
        # Simple FFN
        self.ffn = nn.Linear(d_model, d_model)
        
        # Lag prediction (simple)
        self.lag_net = nn.Linear(d_model, 1)
        
        # Output
        self.out = nn.Linear(d_model, 1)
        
        # Store attention weights
        self.last_attn_weights = None
        
    def forward(self, X_ret, X_vol, target_idx=None):
        B, T, N = X_ret.shape
        device = X_ret.device
        
        # Encode: stack and project
        x = torch.stack([X_ret, X_vol], dim=-1)  # (B, T, N, 2)
        x = self.input_proj(x)  # (B, T, N, D)
        
        # Add embeddings (precompute for speed)
        x = x + self.stock_emb.weight.view(1, 1, N, -1)
        x = x + self.pos_emb.weight.view(1, T, 1, -1)
        
        # Flatten and attend
        x_flat = x.view(B, T * N, -1)
        attn_out, attn_weights = self.attn(x_flat, x_flat, x_flat, need_weights=True)
        self.last_attn_weights = attn_weights.detach()
        
        x_flat = self.norm(x_flat + attn_out)
        x_flat = x_flat + F.relu(self.ffn(x_flat))
        
        # Last timestep
        x = x_flat.view(B, T, N, -1)[:, -1, :, :]  # (B, N, D)
        
        # Lags and prediction
        lags = self.min_lag + torch.sigmoid(self.lag_net(x).squeeze(-1)) * (self.max_lag - self.min_lag)
        
        if target_idx is not None:
            pred = self.out(x[:, target_idx, :])
        else:
            pred = self.out(x).squeeze(-1)
        
        return {
            'predictions': pred,
            'lags': lags.mean(0),
            'attn_weights': self.last_attn_weights
        }
    
    def get_causal_strengths(self, X_ret, X_vol, target_idx):
        """
        Extract causal strengths from attention weights.
        
        Returns:
            strengths: (N,) tensor - how much each stock influences the target
            lags: (N,) tensor - predicted lag for each stock
        """
        B, T, N = X_ret.shape
        
        with torch.no_grad():
            out = self.forward(X_ret, X_vol, target_idx)
            
            # Attention weights: (B, T*N, T*N)
            # We want: how much does stock j at any time influence stock target_idx at time T?
            attn = out['attn_weights']  # (B, T*N, T*N)
            
            # Target position in flattened tensor: last timestep, target stock
            # Position = (T-1) * N + target_idx
            target_pos = (T - 1) * N + target_idx
            
            # Get attention FROM target TO all positions
            target_attn = attn[:, target_pos, :]  # (B, T*N)
            
            # Reshape to (B, T, N) and sum over time to get per-stock influence
            target_attn = target_attn.view(B, T, N)
            
            # Sum attention over time dimension -> (B, N)
            stock_influence = target_attn.sum(dim=1)  # (B, N)
            
            # Average over batch -> (N,)
            causal_strengths = stock_influence.mean(dim=0)
            
            # Normalize to [0, 1]
            causal_strengths = causal_strengths / (causal_strengths.sum() + 1e-8)
            
            return causal_strengths.cpu().numpy(), out['lags'].cpu().numpy()


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


def train_stock(idx, name, data, config, epochs=10, batch_size=1024, verbose=True):
    """Train one stock - ULTRA FAST."""
    device = config.DEVICE
    
    if verbose:
        print(f"\n{'─'*50}")
        print(f"📈 Training: {name} ({idx+1}/{data['n_stocks']})")
        print(f"{'─'*50}")
    
    # Use subset of training data for speed (every 2nd sample)
    X_ret, X_vol, y = data['train']
    step = 2  # Use half the data
    train_ds = TensorDataset(
        torch.FloatTensor(X_ret[::step]),
        torch.FloatTensor(X_vol[::step]),
        torch.FloatTensor(y[::step, idx])
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Smaller validation set
    X_ret_v, X_vol_v, y_v = data['val']
    val_size = min(5000, len(X_ret_v))
    val_ds = TensorDataset(
        torch.FloatTensor(X_ret_v[:val_size]),
        torch.FloatTensor(X_vol_v[:val_size]),
        torch.FloatTensor(y_v[:val_size, idx])
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    # Smaller, faster model
    model = AttentionCausalModel(
        n_stocks=data['n_stocks'],
        lookback=config.LOOKBACK_WINDOW,
        d_model=16,  # Smaller
        n_heads=1,   # Single head
        max_lag=12
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)  # Higher LR
    
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
        
        # Print epoch progress
        if verbose:
            status = "✓ best" if val_loss < best_loss else ""
            print(f"  Epoch {epoch+1:2d}: train={train_loss:.5f} val={val_loss:.5f} R²={r2:.4f} {status}")
        
        # Early stopping (patience = 1 for speed)
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 1:  # Very aggressive
                if verbose:
                    print(f"  ⏹ Early stop @ epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_state)
    model.to(device)
    
    # Extract causal strengths from ATTENTION WEIGHTS
    model.eval()
    X_r = torch.FloatTensor(data['val'][0][:256]).to(device)
    X_v = torch.FloatTensor(data['val'][1][:256]).to(device)
    
    causal_strengths, lags = model.get_causal_strengths(X_r, X_v, idx)
    
    # Show top influencers
    if verbose:
        top_k = 5
        sorted_idx = np.argsort(causal_strengths)[::-1]
        print(f"\n  Top {top_k} influencers on {name}:")
        for rank, j in enumerate(sorted_idx[:top_k]):
            source = data['stock_names'][j]
            strength = causal_strengths[j]
            lag = lags[j]
            print(f"    {rank+1}. {source:6s} → {name}: strength={strength:.3f}, lag={lag:.1f} intervals ({lag*5:.0f} min)")
    
    return {
        'name': name,
        'r2': max(history['val_r2']),
        'causal_strengths': causal_strengths,  # From attention weights!
        'lags': lags,
        'history': history,
        'epochs_trained': len(history['train_loss'])
    }


def create_final_visualization(all_results, stock_names, save_path='plots/final_results.png'):
    """Create comprehensive final visualization."""
    os.makedirs('plots', exist_ok=True)
    
    n_stocks = len(all_results)
    
    fig = plt.figure(figsize=(20, 16), facecolor='#0a0a0a')
    
    # 1. R² scores
    ax1 = fig.add_subplot(2, 2, 1, facecolor='#1a1a2e')
    r2_scores = [r['r2'] for r in all_results]
    names = [r['name'] for r in all_results]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(r2_scores)))
    bars = ax1.barh(names, r2_scores, color=colors)
    ax1.set_xlabel('R² Score', color='white', fontsize=12)
    ax1.set_title('Prediction Performance (R²)', color='white', fontsize=14, fontweight='bold')
    ax1.tick_params(colors='white')
    for bar, score in zip(bars, r2_scores):
        ax1.text(score + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', color='white', fontsize=9)
    
    # 2. Causal strength heatmap (FROM ATTENTION)
    ax2 = fig.add_subplot(2, 2, 2, facecolor='#1a1a2e')
    strength_matrix = np.zeros((n_stocks, n_stocks))
    for i, r in enumerate(all_results):
        strength_matrix[i, :] = r['causal_strengths']
    
    im = ax2.imshow(strength_matrix, cmap='hot', aspect='auto')
    ax2.set_xticks(range(n_stocks))
    ax2.set_yticks(range(n_stocks))
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8, color='white')
    ax2.set_yticklabels(names, fontsize=8, color='white')
    ax2.set_title('Causal Strength (from Attention Weights)', color='white', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Source Stock', color='white')
    ax2.set_ylabel('Target Stock', color='white')
    plt.colorbar(im, ax=ax2, label='Attention-based Causal Strength')
    
    # 3. Lag distribution
    ax3 = fig.add_subplot(2, 2, 3, facecolor='#1a1a2e')
    all_lags = []
    for r in all_results:
        all_lags.extend(r['lags'].tolist())
    ax3.hist(all_lags, bins=20, color='#3498db', edgecolor='white', alpha=0.7)
    mean_lag = np.mean(all_lags)
    ax3.axvline(mean_lag, color='#e74c3c', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_lag:.1f} intervals ({mean_lag*5:.0f} min)')
    ax3.set_xlabel('Lag (5-min intervals)', color='white', fontsize=12)
    ax3.set_ylabel('Frequency', color='white', fontsize=12)
    ax3.set_title('Distribution of Learned Lags', color='white', fontsize=14, fontweight='bold')
    ax3.tick_params(colors='white')
    ax3.legend(facecolor='#2d2d44', labelcolor='white')
    
    # 4. Top causal relationships
    ax4 = fig.add_subplot(2, 2, 4, facecolor='#1a1a2e')
    
    # Collect top relationships
    relationships = []
    for i, r in enumerate(all_results):
        for j, source in enumerate(stock_names):
            if i != j and r['causal_strengths'][j] > 0.05:
                relationships.append({
                    'source': source,
                    'target': r['name'],
                    'strength': r['causal_strengths'][j],
                    'lag': r['lags'][j]
                })
    
    # Sort and take top 15
    relationships = sorted(relationships, key=lambda x: x['strength'], reverse=True)[:15]
    
    if relationships:
        labels = [f"{r['source']}→{r['target']}" for r in relationships]
        strengths = [r['strength'] for r in relationships]
        lags = [r['lag'] for r in relationships]
        
        colors = plt.cm.coolwarm(np.array(lags) / 12)
        bars = ax4.barh(range(len(labels)), strengths, color=colors)
        ax4.set_yticks(range(len(labels)))
        ax4.set_yticklabels(labels, fontsize=9, color='white')
        ax4.set_xlabel('Causal Strength (Attention)', color='white', fontsize=12)
        ax4.set_title('Top 15 Causal Relationships', color='white', fontsize=14, fontweight='bold')
        ax4.tick_params(colors='white')
        
        # Add lag annotations
        for bar, lag in zip(bars, lags):
            ax4.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{lag:.0f}i', va='center', color='#888', fontsize=8)
    
    plt.suptitle('🎯 Causal Volatility Transmission (Attention-Based)', 
                 fontsize=18, fontweight='bold', color='white', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    print(f"✓ Visualization saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()
    
    config = Config()
    tickers = [t.strip().upper() for t in args.tickers.split(',')] if args.tickers else None
    
    print("\n" + "="*60)
    print("🚀 ULTRA-FAST CAUSAL DISCOVERY")
    print("   Attention weights = Causal strength")
    print("="*60)
    print(f"  Batch: {args.batch_size} | Epochs: {args.epochs} | Patience: 1")
    print(f"  Model: d=16, heads=1 | Data: 50% subsample")
    print("="*60 + "\n")
    
    # Load data
    data = load_data(config, tickers)
    print(f"Loaded {data['n_stocks']} stocks\n")
    
    # Train all stocks
    results = []
    start = time.time()
    
    for idx, name in enumerate(data['stock_names']):
        r = train_stock(idx, name, data, config, args.epochs, args.batch_size, verbose=True)
        results.append(r)
        print(f"  ✅ {name} complete: R²={r['r2']:.4f} ({r['epochs_trained']} epochs)\n")
    
    elapsed = time.time() - start
    
    # Summary
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Avg R²: {np.mean([r['r2'] for r in results]):.4f}")
    print(f"{'='*60}\n")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    rows = []
    for r in results:
        for j, source in enumerate(data['stock_names']):
            if r['causal_strengths'][j] > 0.03:  # Threshold
                rows.append({
                    'source': source,
                    'target': r['name'],
                    'causal_strength': float(r['causal_strengths'][j]),
                    'lag_intervals': float(r['lags'][j]),
                    'lag_minutes': float(r['lags'][j]) * 5
                })
    
    df = pd.DataFrame(rows).sort_values('causal_strength', ascending=False)
    df.to_csv('results/causal_relationships.csv', index=False)
    print(f"✓ Saved {len(df)} relationships to results/causal_relationships.csv")
    
    # Final visualization
    create_final_visualization(results, data['stock_names'])
    
    # Training summary
    summary = pd.DataFrame([{
        'stock': r['name'], 
        'r2': r['r2'], 
        'epochs': r['epochs_trained']
    } for r in results])
    summary.to_csv(f'results/training_{datetime.now():%Y%m%d_%H%M}.csv', index=False)


if __name__ == '__main__':
    main()
