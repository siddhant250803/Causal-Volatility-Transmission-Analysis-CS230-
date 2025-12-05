# Causal Volatility Transmission via Attention

**CS230 Deep Learning Project** — Discovering lead/lag relationships in stock volatility using attention mechanisms.

## Overview

This project uses **self-attention** to discover causal relationships between stocks' volatility. The key insight: **attention weights directly represent causal influence strength** — no separate "causal gates" needed.

Given 5-minute returns data for multiple stocks, the model learns:
1. **Which stocks influence which** (from attention weights)
2. **How strong the influence is** (attention magnitude)  
3. **The time delay of influence** (learned lag in 5-min intervals)

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train on 23 stocks across Tech, Energy, Finance, Healthcare, Consumer sectors
python scripts/fast_train.py --tickers "AAPL,MSFT,NVDA,AMD,INTC,ORCL,CSCO,XOM,CVX,COP,SLB,VLO,OXY,HAL,JPM,BAC,WFC,JNJ,PFE,PG,KO,BA,CAT" --epochs 6
```

## Model Architecture

```
Input: Returns + Volatility (B, T=12, N stocks)
              ↓
    ┌─────────────────────┐
    │  Joint Encoder      │  Linear: ℝ² → ℝᵈ
    └─────────────────────┘
              ↓
    ┌─────────────────────┐
    │  Stock + Position   │  Learnable embeddings
    │  Embeddings         │
    └─────────────────────┘
              ↓
    ┌─────────────────────┐
    │  Self-Attention     │  ← Attention weights αᵢⱼ
    │  (Causal Mechanism) │    = causal strength
    └─────────────────────┘
              ↓
    ┌─────────────────────┐
    │  FFN + LayerNorm    │
    └─────────────────────┘
              ↓
    ┌─────────────────────┐
    │  Temporal Pooling   │  Last timestep
    └─────────────────────┘
          ↓       ↓
    ┌─────────┐ ┌─────────┐
    │ Lag Net │ │ Output  │
    │ ℓⱼ∈[2,12]│ │ σ̂ᵢ,ₜ₊₁  │
    └─────────┘ └─────────┘
```

**Key Design Choice:** Attention weights ARE the causal strengths. When stock A attends strongly to stock B, it means B's volatility causally influences A's future volatility.

## Output

### Visualizations (`plots/final_results.png`)

| Panel | Description |
|-------|-------------|
| **R² Scores** | Prediction performance per stock |
| **Causal Heatmap** | N×N matrix of attention-based causal strengths |
| **Lag Distribution** | Histogram of learned time delays (10-60 min) |
| **Top Relationships** | Strongest causal pairs with lag annotations |

### CSV Results (`results/`)

- `causal_relationships.csv` — All significant causal pairs:
  ```
  source, target, causal_strength, lag_intervals, lag_minutes
  NVDA,   AAPL,   0.152,           3,             15
  XOM,    CVX,    0.098,           5,             25
  ```

## Project Structure

```
CS230/
├── scripts/
│   └── fast_train.py       # Main training script
├── src/
│   ├── config.py           # Hyperparameters
│   ├── data/
│   │   └── dataloader.py   # Data loading & preprocessing
│   ├── models/
│   │   └── efficient_attention.py
│   └── utils/
│       ├── losses.py
│       └── metrics.py
├── results/                # Output CSVs
├── plots/                  # Visualizations
├── checkpoints/            # Saved models
└── paper/                  # LaTeX report
```

## Method Details

### Data Preprocessing
1. Load 5-minute stock returns
2. Compute realized volatility: σₜ = √(Σᵢ rₜ₋ᵢ²) over 1-hour rolling window
3. Z-score normalize using training set statistics
4. Create sequences with lookback T=12 (1 hour)

### Training
- **Loss:** MSE on volatility prediction
- **Optimizer:** AdamW (lr=0.005, weight_decay=0.01)
- **Early stopping:** Patience=1 epoch
- **Batch size:** 1024

### Causal Extraction
After training, extract attention weights α from the model:
- αᵢⱼ = how much stock j influences stock i
- Normalize across sources for each target
- Threshold at 0.03 for significance

## Interpretation

| Lag (intervals) | Time Delay | Interpretation |
|-----------------|------------|----------------|
| 2 | 10 min | Fast information transmission |
| 6 | 30 min | Medium-term spillover |
| 12 | 60 min | Slow regime effects |

**Example findings:**
- Tech stocks (NVDA, AMD) often lead other tech stocks by 15-25 min
- Energy sector shows strong within-sector causality (XOM→CVX)
- Financial stocks (JPM, BAC) exhibit tight coupling with short lags

## Citation

```bibtex
@article{cs230volatility,
  title={Causal Volatility Transmission via Attention},
  author={Sukhani, S. and Roger, A. and Alhusseini, M.},
  journal={CS230 Deep Learning Project},
  institution={Stanford University},
  year={2025}
}
```

Data: Pelger (2020) "Understanding Systematic Risk: A High-Frequency Approach"
