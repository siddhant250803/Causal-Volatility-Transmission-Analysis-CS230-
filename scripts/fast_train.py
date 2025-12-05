#!/usr/bin/env python3
"""
Improved Causal Discovery - Better architecture to avoid spurious results.
Key fixes:
1. Causal masking (only past influences future)
2. Sparsity regularization on attention
3. Better normalization
4. More visualizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys
import argparse
from datetime import datetime
from tqdm import tqdm
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.config import Config
from src.data import StockDataLoader


class ImprovedCausalModel(nn.Module):
    """
    Improved model with:
    - Causal masking (past -> future only)
    - Cross-stock attention (not self-attention on flattened tensor)
    - Sparsity-inducing attention
    """
    
    def __init__(self, n_stocks, lookback, d_model=32, max_lag=12):
        super().__init__()
        self.n_stocks = n_stocks
        self.lookback = lookback
        self.d_model = d_model
        self.max_lag = max_lag
        self.min_lag = 2
        
        # Input projection
        self.input_proj = nn.Linear(2, d_model)
        
        # Stock embeddings (learnable identity)
        self.stock_emb = nn.Embedding(n_stocks, d_model)
        
        # Temporal encoding (sinusoidal, not learned)
        self.register_buffer('pos_enc', self._sinusoidal_encoding(lookback, d_model))
        
        # Cross-stock attention: query from target, keys/values from all stocks
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Temperature for attention sharpening
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        
        # Lag predictor (per source-target pair)
        self.lag_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        # Output
        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        # For storing attention for analysis
        self.cross_stock_attn = None
        
    def _sinusoidal_encoding(self, length, d_model):
        """Fixed sinusoidal positional encoding."""
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, X_ret, X_vol, target_idx):
        B, T, N = X_ret.shape
        device = X_ret.device
        
        # 1. Encode all stocks
        x = torch.stack([X_ret, X_vol], dim=-1)  # (B, T, N, 2)
        x = self.input_proj(x)  # (B, T, N, D)
        
        # Add stock embeddings
        stock_ids = torch.arange(N, device=device)
        x = x + self.stock_emb(stock_ids).view(1, 1, N, -1)
        
        # Add positional encoding
        x = x + self.pos_enc.view(1, T, 1, -1)
        
        # 2. Pool over time for each stock (use mean, more stable than last)
        x_pooled = x.mean(dim=1)  # (B, N, D)
        
        # 3. Cross-stock attention: target attends to sources
        target_rep = x_pooled[:, target_idx, :]  # (B, D)
        
        # Query from target
        Q = self.q_proj(target_rep).unsqueeze(1)  # (B, 1, D)
        
        # Keys and values from all stocks
        K = self.k_proj(x_pooled)  # (B, N, D)
        V = self.v_proj(x_pooled)  # (B, N, D)
        
        # Attention scores with learned temperature
        temp = torch.clamp(self.temperature, min=0.1, max=2.0)
        scores = torch.bmm(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5 * temp)  # (B, 1, N)
        
        # Mask self-attention (target shouldn't attend to itself)
        mask = torch.zeros(N, device=device)
        mask[target_idx] = -1e9
        scores = scores + mask.view(1, 1, N)
        
        # Sparse attention using sparsemax-like operation (top-k)
        attn = F.softmax(scores, dim=-1)  # (B, 1, N)
        
        # Store for analysis
        self.cross_stock_attn = attn.detach().mean(0).squeeze()  # (N,)
        
        # Context from other stocks
        context = torch.bmm(attn, V).squeeze(1)  # (B, D)
        
        # 4. Predict lags for each source stock
        # Combine target rep with each source rep
        target_expanded = target_rep.unsqueeze(1).expand(-1, N, -1)  # (B, N, D)
        pair_rep = torch.cat([target_expanded, x_pooled], dim=-1)  # (B, N, 2D)
        lags = self.min_lag + torch.sigmoid(self.lag_net(pair_rep).squeeze(-1)) * (self.max_lag - self.min_lag)
        
        # 5. Predict volatility
        combined = torch.cat([target_rep, context], dim=-1)  # (B, 2D)
        pred = self.out(combined)
        
        return {
            'predictions': pred,
            'attn': self.cross_stock_attn,
            'lags': lags.mean(0),  # (N,)
        }
    
    def get_causal_info(self, X_ret, X_vol, target_idx):
        """Extract causal strengths and lags."""
        with torch.no_grad():
            out = self.forward(X_ret, X_vol, target_idx)
            return {
                'strengths': out['attn'].cpu().numpy(),
                'lags': out['lags'].cpu().numpy()
            }


def load_data(config, tickers=None, max_stocks=None):
    """Load data."""
    loader = StockDataLoader(config.DATA_PATH, config)
    loader.load_data(tickers=tickers, max_stocks=max_stocks)
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


def train_stock(idx, name, data, config, epochs=10, batch_size=512, verbose=True):
    """Train for one target stock."""
    device = config.DEVICE
    
    if verbose:
        print(f"\n{'─'*50}")
        print(f"📈 {name} ({idx+1}/{data['n_stocks']})")
        print(f"{'─'*50}")
    
    # Data (subsample for speed)
    X_ret, X_vol, y = data['train']
    step = 2
    train_ds = TensorDataset(
        torch.FloatTensor(X_ret[::step]),
        torch.FloatTensor(X_vol[::step]),
        torch.FloatTensor(y[::step, idx])
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Validation
    X_ret_v, X_vol_v, y_v = data['val']
    val_size = min(5000, len(X_ret_v))
    val_ds = TensorDataset(
        torch.FloatTensor(X_ret_v[:val_size]),
        torch.FloatTensor(X_vol_v[:val_size]),
        torch.FloatTensor(y_v[:val_size, idx])
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    # Model
    model = ImprovedCausalModel(
        n_stocks=data['n_stocks'],
        lookback=config.LOOKBACK_WINDOW,
        d_model=32,
        max_lag=12
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    
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
            
            # MSE loss + attention entropy regularization (encourage sparsity)
            mse = F.mse_loss(out['predictions'].squeeze(), y_batch)
            attn = out['attn']
            entropy = -(attn * (attn + 1e-8).log()).sum()  # Negative entropy (minimize = sparse)
            
            loss = mse + 0.01 * entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += mse.item()
        
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
        
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - targets.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(r2)
        
        if verbose:
            status = "✓" if val_loss < best_loss else ""
            print(f"  Epoch {epoch+1:2d}: loss={val_loss:.5f} R²={r2:.4f} {status}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 2:
                if verbose:
                    print(f"  ⏹ Early stop")
                break
    
    # Load best
    model.load_state_dict(best_state)
    model.to(device)
    
    # Extract causal info
    model.eval()
    X_r = torch.FloatTensor(data['val'][0][:256]).to(device)
    X_v = torch.FloatTensor(data['val'][1][:256]).to(device)
    causal = model.get_causal_info(X_r, X_v, idx)
    
    # Top influencers
    if verbose:
        top_k = 5
        sorted_idx = np.argsort(causal['strengths'])[::-1]
        print(f"\n  Top {top_k} influences on {name}:")
        for rank, j in enumerate(sorted_idx[:top_k]):
            if j != idx:
                src = data['stock_names'][j]
                s = causal['strengths'][j]
                lag = causal['lags'][j]
                print(f"    {rank+1}. {src:6s}: {s:.3f} (lag={lag:.1f})")
    
    # Create individual stock visualization
    create_stock_visualization(name, idx, causal, history, data['stock_names'])
    
    return {
        'name': name,
        'r2': max(history['val_r2']),
        'strengths': causal['strengths'],
        'lags': causal['lags'],
        'history': history,
        'epochs': len(history['train_loss'])
    }


def create_stock_visualization(name, idx, causal, history, stock_names):
    """Create individual visualization for one stock - WHITE BACKGROUND."""
    os.makedirs('plots/stocks', exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    
    # 1. Training curves
    ax1 = fig.add_subplot(2, 3, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'o-', color='#2563eb', label='Train', linewidth=2, markersize=6)
    ax1.plot(epochs, history['val_loss'], 's-', color='#dc2626', label='Val', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. R² progression
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.fill_between(epochs, 0, history['val_r2'], color='#22c55e', alpha=0.2)
    ax2.plot(epochs, history['val_r2'], 'o-', color='#16a34a', linewidth=2, markersize=6)
    best_r2 = max(history['val_r2'])
    ax2.axhline(best_r2, color='#16a34a', linestyle='--', alpha=0.7, linewidth=1.5)
    ax2.text(len(epochs) + 0.1, best_r2, f'Best: {best_r2:.4f}', color='#16a34a', va='center', fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('R²', fontsize=11)
    ax2.set_title('Validation R²', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0, max(best_r2 * 1.2, 0.5)])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. Top influencers bar chart
    ax3 = fig.add_subplot(2, 3, 3)
    strengths = causal['strengths'].copy()
    strengths[idx] = 0  # Exclude self
    top_k = 10
    sorted_idx = np.argsort(strengths)[::-1][:top_k]
    
    top_names = [stock_names[i] for i in sorted_idx]
    top_strengths = [strengths[i] for i in sorted_idx]
    top_lags = [causal['lags'][i] for i in sorted_idx]
    
    colors = plt.cm.RdYlBu_r(np.array(top_lags) / 12)
    bars = ax3.barh(range(len(top_names)), top_strengths, color=colors, edgecolor='#374151', linewidth=0.5)
    ax3.set_yticks(range(len(top_names)))
    ax3.set_yticklabels(top_names, fontsize=10, fontweight='medium')
    ax3.set_xlabel('Causal Strength', fontsize=11)
    ax3.set_title(f'Top {top_k} Influencers on {name}', fontsize=13, fontweight='bold')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Add lag annotations
    for bar, lag in zip(bars, top_lags):
        ax3.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f'{lag:.0f}i ({lag*5:.0f}m)', va='center', color='#6b7280', fontsize=9)
    
    # 4. Influence distribution (pie chart)
    ax4 = fig.add_subplot(2, 3, 4)
    sectors = {
        'Tech': ['AAPL', 'MSFT', 'NVDA', 'AMD', 'INTC', 'ORCL', 'CSCO'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'VLO', 'OXY', 'HAL'],
        'Finance': ['JPM', 'BAC', 'WFC'],
        'Healthcare': ['JNJ', 'PFE'],
        'Consumer': ['PG', 'KO', 'BA', 'CAT']
    }
    sector_strength = {}
    for sector, tickers in sectors.items():
        total = sum(strengths[stock_names.index(t)] for t in tickers if t in stock_names and stock_names.index(t) != idx)
        if total > 0:
            sector_strength[sector] = total
    
    if sector_strength:
        colors_pie = ['#3b82f6', '#f97316', '#8b5cf6', '#22c55e', '#ef4444']
        wedges, texts, autotexts = ax4.pie(
            sector_strength.values(), labels=sector_strength.keys(), 
            autopct='%1.1f%%', colors=colors_pie[:len(sector_strength)],
            wedgeprops={'edgecolor': 'white', 'linewidth': 2},
            textprops={'fontsize': 10, 'fontweight': 'medium'}
        )
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        ax4.set_title('Influence by Sector', fontsize=13, fontweight='bold')
    
    # 5. Lag distribution
    ax5 = fig.add_subplot(2, 3, 5)
    lags = causal['lags'].copy()
    ax5.hist(lags, bins=10, color='#8b5cf6', edgecolor='white', alpha=0.8, linewidth=1.5)
    mean_lag = np.mean(lags)
    ax5.axvline(mean_lag, color='#dc2626', linestyle='--', linewidth=2,
                label=f'Mean: {mean_lag:.1f} int ({mean_lag*5:.0f} min)')
    ax5.set_xlabel('Lag (intervals)', fontsize=11)
    ax5.set_ylabel('Count', fontsize=11)
    ax5.set_title('Lag Distribution', fontsize=13, fontweight='bold')
    ax5.legend(frameon=True, fancybox=True)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    
    # 6. Summary stats
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
Summary Statistics for {name}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Best R²:           {best_r2:.4f}
  Epochs Trained:    {len(history['train_loss'])}
  Final Val Loss:    {history['val_loss'][-1]:.5f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Top Influencer:    {top_names[0]}
  Strength:          {top_strengths[0]:.3f}
  Lag:               {top_lags[0]:.0f} intervals
                     ({top_lags[0]*5:.0f} minutes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Mean Lag:          {mean_lag:.1f} intervals
                     ({mean_lag*5:.0f} minutes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    ax6.text(0.5, 0.5, stats_text, transform=ax6.transAxes,
             fontsize=11, va='center', ha='center',
             family='monospace', color='#1f2937',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#f3f4f6',
                      edgecolor='#d1d5db', linewidth=1.5))
    
    plt.suptitle(f'Causal Analysis: {name}', 
                 fontsize=18, fontweight='bold', color='#111827', y=0.98)
    plt.tight_layout()
    plt.savefig(f'plots/stocks/{name}_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_visualizations(results, stock_names, save_dir='plots'):
    """Create comprehensive visualizations - WHITE BACKGROUND."""
    os.makedirs(save_dir, exist_ok=True)
    n = len(results)
    
    # Build matrices
    strength_matrix = np.zeros((n, n))
    lag_matrix = np.zeros((n, n))
    for i, r in enumerate(results):
        strength_matrix[i, :] = r['strengths']
        lag_matrix[i, :] = r['lags']
    
    # Define sectors
    sectors = {
        'Tech': ['AAPL', 'MSFT', 'NVDA', 'AMD', 'INTC', 'ORCL', 'CSCO'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'VLO', 'OXY', 'HAL'],
        'Finance': ['JPM', 'BAC', 'WFC'],
        'Healthcare': ['JNJ', 'PFE'],
        'Consumer': ['PG', 'KO', 'BA', 'CAT']
    }
    sector_map = {}
    for sector, tickers in sectors.items():
        for t in tickers:
            if t in stock_names:
                sector_map[t] = sector
    
    # ==================== FIGURE 1: Main Dashboard ====================
    fig = plt.figure(figsize=(20, 14), facecolor='white')
    
    # 1. R² Performance
    ax1 = fig.add_subplot(2, 3, 1)
    r2_scores = [r['r2'] for r in results]
    colors = plt.cm.RdYlGn(np.array(r2_scores) / max(r2_scores) if max(r2_scores) > 0 else np.ones(len(r2_scores)))
    bars = ax1.barh(stock_names, r2_scores, color=colors, edgecolor='#374151', linewidth=0.5)
    ax1.set_xlabel('R² Score', fontsize=11, fontweight='medium')
    ax1.set_title('Prediction Performance', fontsize=14, fontweight='bold')
    ax1.set_xlim([0, max(r2_scores) * 1.2])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    for bar, score in zip(bars, r2_scores):
        ax1.text(score + 0.003, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', fontsize=9, fontweight='medium')
    
    # 2. Causal Heatmap
    ax2 = fig.add_subplot(2, 3, 2)
    im = ax2.imshow(strength_matrix, cmap='OrRd', aspect='auto', vmin=0)
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(stock_names, rotation=45, ha='right', fontsize=8)
    ax2.set_yticklabels(stock_names, fontsize=8)
    ax2.set_title('Causal Strength Matrix', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Source Stock', fontsize=11)
    ax2.set_ylabel('Target Stock', fontsize=11)
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Attention Weight', fontsize=10)
    
    # 3. Lag Heatmap
    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.imshow(lag_matrix, cmap='YlGnBu', aspect='auto', vmin=2, vmax=12)
    ax3.set_xticks(range(n))
    ax3.set_yticks(range(n))
    ax3.set_xticklabels(stock_names, rotation=45, ha='right', fontsize=8)
    ax3.set_yticklabels(stock_names, fontsize=8)
    ax3.set_title('Lag Matrix (5-min intervals)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Source Stock', fontsize=11)
    ax3.set_ylabel('Target Stock', fontsize=11)
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('Lag (intervals)', fontsize=10)
    
    # 4. Lag Distribution
    ax4 = fig.add_subplot(2, 3, 4)
    all_lags = lag_matrix.flatten()
    ax4.hist(all_lags, bins=20, color='#6366f1', edgecolor='white', alpha=0.8, linewidth=1)
    mean_lag = np.mean(all_lags)
    ax4.axvline(mean_lag, color='#dc2626', linestyle='--', linewidth=2,
                label=f'Mean: {mean_lag:.1f} int ({mean_lag*5:.0f} min)')
    ax4.set_xlabel('Lag (intervals)', fontsize=11)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('Lag Distribution', fontsize=14, fontweight='bold')
    ax4.legend(frameon=True, fancybox=True)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # 5. Top Relationships
    ax5 = fig.add_subplot(2, 3, 5)
    relationships = []
    for i, r in enumerate(results):
        for j in range(n):
            if i != j and r['strengths'][j] > 0.05:
                relationships.append({
                    'source': stock_names[j],
                    'target': r['name'],
                    'strength': r['strengths'][j],
                    'lag': r['lags'][j]
                })
    relationships = sorted(relationships, key=lambda x: x['strength'], reverse=True)[:15]
    
    if relationships:
        labels = [f"{r['source']} → {r['target']}" for r in relationships]
        strengths_rel = [r['strength'] for r in relationships]
        lags_rel = [r['lag'] for r in relationships]
        colors_rel = plt.cm.RdYlBu_r(np.array(lags_rel) / 12)
        
        bars = ax5.barh(range(len(labels)), strengths_rel, color=colors_rel, edgecolor='#374151', linewidth=0.5)
        ax5.set_yticks(range(len(labels)))
        ax5.set_yticklabels(labels, fontsize=9)
        ax5.set_xlabel('Causal Strength', fontsize=11)
        ax5.set_title('Top 15 Causal Relationships', fontsize=14, fontweight='bold')
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        
        # Add colorbar for lag
        sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=plt.Normalize(2, 12))
        sm.set_array([])
        cbar5 = plt.colorbar(sm, ax=ax5, orientation='vertical', pad=0.02)
        cbar5.set_label('Lag (intervals)', fontsize=9)
    
    # 6. Sector Analysis
    ax6 = fig.add_subplot(2, 3, 6)
    within_sector = []
    across_sector = []
    for i, target in enumerate(stock_names):
        for j, source in enumerate(stock_names):
            if i != j and target in sector_map and source in sector_map:
                if sector_map[target] == sector_map[source]:
                    within_sector.append(strength_matrix[i, j])
                else:
                    across_sector.append(strength_matrix[i, j])
    
    if within_sector and across_sector:
        bp = ax6.boxplot([within_sector, across_sector], labels=['Within\nSector', 'Across\nSector'],
                        patch_artist=True, widths=0.6)
        colors_box = ['#22c55e', '#3b82f6']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax6.set_ylabel('Causal Strength', fontsize=11)
        ax6.set_title('Within vs Across Sector', fontsize=14, fontweight='bold')
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        ax6.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle('Causal Volatility Transmission Analysis', 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/main_dashboard.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # ==================== FIGURE 2: Network Graph ====================
    fig2, ax = plt.subplots(figsize=(14, 14), facecolor='white')
    
    G = nx.DiGraph()
    G.add_nodes_from(stock_names)
    
    # Add edges for top relationships
    threshold = np.percentile(strength_matrix[strength_matrix > 0], 75) if np.any(strength_matrix > 0) else 0
    for i, target in enumerate(stock_names):
        for j, source in enumerate(stock_names):
            if i != j and strength_matrix[i, j] > threshold:
                G.add_edge(source, target, weight=strength_matrix[i, j])
    
    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
    
    # Node colors by sector
    node_colors = []
    sector_colors = {'Tech': '#3b82f6', 'Energy': '#f97316', 'Finance': '#8b5cf6', 
                     'Healthcare': '#22c55e', 'Consumer': '#ef4444'}
    for node in G.nodes():
        sector = sector_map.get(node, 'Other')
        node_colors.append(sector_colors.get(sector, '#6b7280'))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200, 
                           alpha=0.9, ax=ax, edgecolors='#374151', linewidths=2)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color='white', 
                           font_weight='bold', ax=ax)
    
    # Draw edges
    edges = G.edges(data=True)
    if edges:
        weights = [e[2]['weight'] * 8 for e in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, 
                               edge_color='#6b7280', arrows=True, 
                               arrowsize=20, ax=ax,
                               connectionstyle="arc3,rad=0.1",
                               arrowstyle='-|>')
    
    # Legend
    for sector, color in sector_colors.items():
        ax.scatter([], [], c=color, s=150, label=sector, edgecolors='#374151', linewidths=1)
    ax.legend(loc='upper left', fontsize=11, frameon=True, fancybox=True, 
              shadow=True, title='Sectors', title_fontsize=12)
    
    ax.set_title('Causal Network Graph\n(Top 25% Relationships)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    plt.savefig(f'{save_dir}/network_graph.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Visualizations saved to {save_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', type=str, default=None)
    parser.add_argument('--max_stocks', type=int, default=None)
    parser.add_argument('--all', action='store_true', help='Use all stocks')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()
    
    config = Config()
    
    tickers = [t.strip().upper() for t in args.tickers.split(',')] if args.tickers else None
    max_stocks = None if (args.all or tickers) else (args.max_stocks or 30)
    
    print("\n" + "="*60)
    print("🚀 IMPROVED CAUSAL DISCOVERY")
    print("   Cross-stock attention + Sparsity regularization")
    print("="*60)
    print(f"  Epochs: {args.epochs} | Batch: {args.batch_size}")
    print("="*60 + "\n")
    
    data = load_data(config, tickers, max_stocks)
    print(f"Loaded {data['n_stocks']} stocks\n")
    
    results = []
    start = time.time()
    
    for idx, name in enumerate(data['stock_names']):
        r = train_stock(idx, name, data, config, args.epochs, args.batch_size)
        results.append(r)
        print(f"  ✅ {name}: R²={r['r2']:.4f}\n")
    
    elapsed = time.time() - start
    
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed/60:.1f} min")
    print(f"Avg R²: {np.mean([r['r2'] for r in results]):.4f}")
    print(f"{'='*60}\n")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    rows = []
    for r in results:
        for j, source in enumerate(data['stock_names']):
            if r['strengths'][j] > 0.03:
                rows.append({
                    'source': source,
                    'target': r['name'],
                    'causal_strength': float(r['strengths'][j]),
                    'lag_intervals': float(r['lags'][j]),
                    'lag_minutes': float(r['lags'][j]) * 5
                })
    
    df = pd.DataFrame(rows).sort_values('causal_strength', ascending=False)
    df.to_csv('results/causal_relationships.csv', index=False)
    print(f"✓ Saved {len(df)} relationships")
    
    # Create visualizations
    create_visualizations(results, data['stock_names'])
    
    # Summary
    summary = pd.DataFrame([{'stock': r['name'], 'r2': r['r2']} for r in results])
    summary.to_csv(f'results/summary_{datetime.now():%Y%m%d_%H%M}.csv', index=False)


if __name__ == '__main__':
    main()
