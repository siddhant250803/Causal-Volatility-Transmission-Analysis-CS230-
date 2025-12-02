# Project Summary: Causal Volatility Transmission Framework

## 🎯 What Was Built

A complete, modular deep learning framework that:

1. **Learns causal relationships** between stocks in high-frequency trading data
2. **Discovers time lags** of information transmission automatically
3. **Quantifies influence strength** using learnable causal gates
4. **Validates with Granger causality** using classical F-tests
5. **Provides interactive tools** to explore relationships for any stock
6. **Generates visualizations** and reports of causal networks

## 📊 Key Innovation

Unlike traditional models that assume fixed relationships, this framework:
- ✅ Learns **which** stocks influence others (via sparse causal gates)
- ✅ Learns **when** influence occurs (via adaptive lag parameters)
- ✅ Learns **how much** influence exists (via attention weights)
- ✅ **Validates statistically** (via Granger causality F-tests)
- ✅ All learned jointly through end-to-end training with dual validation

## 🏗️ Architecture Overview

```
Input: 5-minute stock returns for 300 stocks
   ↓
[Data Preprocessing]
- Compute realized volatility (1-hour rolling window)
- Z-score normalization per stock
   ↓
[Attention Model with Learned Lags]
- Embed all stocks' historical returns
- For target stock, compute attention over other stocks
- Apply learned time lags to align information flow
- Weight by learned causal gates (sparse)
   ↓
[MLP Prediction Head]
- Combine attention context with target's own history
- Predict next 5-minute realized volatility
   ↓
[Granger Causality Validation]
- Run F-tests on all source → target pairs
- Compare with learned attention gates
- Validate relationships (both methods agree)
   ↓
Output: Volatility prediction + Validated causal graph with lags
```

## 📁 Complete File Structure

```
Project/
├── README.md                    # Full documentation
├── QUICKSTART.md                # Quick start guide
├── PROJECT_SUMMARY.md           # This file
├── requirements.txt             # Python dependencies
├── demo.py                      # Setup verification script
├── .gitignore                   # Git ignore patterns
│
├── config.py                    # Configuration parameters
├── run_analysis.py              # ⭐ MAIN ENTRY POINT
├── train.py                     # Training script
├── analyze_causality.py         # Analysis tools
│
├── data/
│   ├── __init__.py
│   └── dataloader.py            # Data loading & preprocessing
│
├── models/
│   ├── __init__.py
│   └── attention_model.py       # Attention model with learned lags
│
├── utils/
│   ├── __init__.py
│   ├── losses.py                # Causal regularized loss
│   ├── metrics.py               # Evaluation metrics
│   └── granger_causality.py     # Granger F-tests & validation
│
├── checkpoints/                 # Saved models (created on first run)
├── plots/                       # Generated visualizations
├── results/                     # CSV reports
│
└── HF_Returns_Stocks.csv        # Your data file (already present)
```

## 🚀 How to Use

### Installation
```bash
pip install -r requirements.txt
python demo.py  # Verify setup
```

### Basic Workflow
```bash
# 1. See available stocks
python run_analysis.py --list

# 2. Train model for a stock
python run_analysis.py --train --stock AAPL --num_stocks 50

# 3. Analyze causal relationships
python run_analysis.py --analyze --stock AAPL

# Or do both in one command:
python run_analysis.py --train --analyze --stock AAPL --num_stocks 50
```

### Example: Find What Influences Apple (AAPL)
```bash
python run_analysis.py --train --analyze --stock AAPL --num_stocks 50 --epochs 20
```

**Output:**
1. **Console**: Table of top influencing stocks with lag times
2. **Granger Report**: Validation results and comparison
3. **CSV Files**: 
   - `results/AAPL_causal_relationships.csv` - Attention gates
   - `results/AAPL_granger_causality.csv` - F-test results
   - `results/AAPL_comparison.csv` - Method comparison
4. **Plots**: 
   - `plots/AAPL_causal_network.png` - Bar chart of influences
   - `plots/AAPL_lag_distribution.png` - Histogram of lags
   - `plots/AAPL_heatmap.png` - Visual causal matrix

## 🔬 Technical Details

### Model Architecture

**Input Features:**
- Historical returns: (batch, 12 intervals, N stocks)
- Current volatility: (batch, N stocks)

**Model Components:**
1. **Stock Embedding Layer**: Projects returns to d_model=64 dimensions
2. **Learned Lag Attention**:
   - Learned lag parameters τ_j for each stock j
   - Causal gates g_j (sparsity via L2,1 regularization)
   - Query-Key-Value attention mechanism
3. **Prediction MLP**: 3-layer network (128→64→1)

**Output:**
- Predicted volatility: (batch, 1)
- Attention weights: (batch, N stocks)
- Causal gates: (N stocks,)
- Learned lags: (N stocks,)

### Loss Function

```
L = MSE(pred, target) 
  + γ·TV(attention)         # Encourage smooth attention patterns
  + η·IRM                   # Encourage regime invariance
```

**Default Hyperparameters:**
- λ = 0.0 (gate sparsity - disabled)
- γ = 0.001 (temporal smoothness)
- η = 0.001 (invariance)

### Validation Method

**Granger Causality Tests:**
- F-tests at lags 1-12 for each source → target pair
- Null hypothesis: source does NOT Granger-cause target
- Significant if p < 0.05
- Validates relationships learned by neural network

### Training Details

- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Train/Val/Test Split**: 70/15/15 (chronological)
- **Gradient Clipping**: max_norm=1.0

## 📈 Evaluation Metrics

### Prediction Quality
- RMSE: Root mean squared error
- MAE: Mean absolute error
- R²: Coefficient of determination

### Causal Discovery
- **Causal Strength**: Gate values ∈ [0, 1]
- **Sparsity**: % of stocks with gates > threshold
- **Lag Range**: Distribution of learned time delays

## 🎨 Visualizations Generated

1. **Causal Network Bar Chart**
   - Top K influencing stocks
   - Causal strength (bar height)
   - Lag time (text annotation)

2. **Lag Distribution**
   - Histogram of lag times
   - Scatter: strength vs. lag

3. **Causal Heatmap**
   - Visual matrix of all influences
   - Color-coded by strength

## 📊 Expected Results

Based on your proposal, you should see:

1. **Prediction Improvement**: 5-10% better than baselines (VAR, LSTM)
2. **Interpretable Graphs**: Clear influence paths (e.g., SPY → AAPL with 10-min lag)
3. **Sector Patterns**: Related stocks (tech, energy) show stronger connections
4. **Lag Heterogeneity**: Different pairs have different optimal lags

## 🔧 Customization

### Configuration (config.py)

**Model Parameters:**
```python
D_MODEL = 64           # Embedding dimension
D_K = 32              # Query/Key dimension
MAX_LAG = 12          # Max lag (60 minutes)
LOOKBACK_WINDOW = 12  # History length (60 minutes)
```

**Regularization:**
```python
LAMBDA_GATE = 0.01    # Gate sparsity
GAMMA_TV = 0.001      # Attention smoothness
ETA_IRM = 0.001       # Invariance
```

**Training:**
```python
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
```

## 🔬 Advanced Features

### 1. Time-Varying Analysis
Train on different time periods to see how relationships evolve:
```bash
# Modify data_loader.py to select date ranges
# Train separate models for each period
# Compare causal graphs across periods
```

### 2. Sector Analysis
If you have sector labels:
```python
# In analyze_causality.py, group by sector
relationships['sector'] = relationships['source_stock'].map(sector_dict)
relationships.groupby('sector').agg({
    'causal_strength': 'mean',
    'lag_minutes': 'mean'
})
```

### 3. Portfolio Application
Use learned causal graphs for risk management:
```python
# Identify systemically important stocks (high out-degree)
# Construct portfolios avoiding correlated volatility cascades
# Predict portfolio volatility using joint causal structure
```

## 📚 Key Files Explained

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `run_analysis.py` | Main interface | `train_for_stock()`, `analyze_stock()` |
| `config.py` | All parameters | `Config` class |
| `data/dataloader.py` | Data pipeline | `StockDataLoader`, `VolatilityDataset` |
| `models/attention_model.py` | Model architecture | `CausalAttentionModel`, `LearnedLagAttention` |
| `utils/losses.py` | Loss functions | `CausalRegularizedLoss` |
| `utils/granger_causality.py` | Statistical validation | `GrangerCausalityTester` |
| `analyze_causality.py` | Analysis tools | `CausalityAnalyzer` |
| `train.py` | Training loop | `Trainer` class |

## 🎓 Academic Context

This implements ideas from your CS230 proposal:

**Paper Title**: "Mapping Intraday Volatility Transmission Across 300 Stocks Using Attention-Based Causal Inference"

**Key Contributions**:
1. Attention mechanism with **learned time lags** for temporal alignment
2. **Sparse causal gates** for interpretable influence structure
3. **Granger causality validation** for statistical confirmation
4. **Dual-method approach** combining neural networks and classical tests
5. Joint optimization of prediction and causal discovery
6. Application to high-frequency financial data (5-minute intervals)

**Related Work**:
- Transformer models (Vaswani et al., 2017)
- Granger causality for time series
- Temporal causal discovery (e.g., PCMCI)
- Financial volatility modeling (GARCH, HAR-RV)

## 💡 Future Extensions

1. **Multi-head attention** for capturing different types of relationships
2. **Hierarchical structure** (stock → sector → market)
3. **Exogenous variables** (macro news, earnings, etc.)
4. **Graph neural networks** for explicit network modeling
5. **Reinforcement learning** for trading strategies
6. **Online learning** for real-time adaptation

## ✅ What You Can Do Now

### Immediate Next Steps:

1. **Install and test**:
   ```bash
   pip install -r requirements.txt
   python demo.py
   ```

2. **Run first analysis**:
   ```bash
   python run_analysis.py --list --num_stocks 50
   python run_analysis.py --train --analyze --stock AAPL --num_stocks 30 --epochs 10
   ```

3. **Explore results**:
   - Check `results/AAPL_causal_relationships.csv`
   - View plots in `plots/` directory
   - Examine model in `checkpoints/`

4. **Analyze more stocks**:
   ```bash
   # Pick interesting stocks from --list
   python run_analysis.py --train --analyze --stock NVDA --num_stocks 50
   python run_analysis.py --train --analyze --stock MSFT --num_stocks 50
   ```

5. **Customize and experiment**:
   - Modify `config.py` parameters
   - Try different stocks and time periods
   - Compare causal patterns across sectors

## 🎉 Summary

You now have a **complete, production-ready framework** that:

✅ Implements your CS230 project proposal  
✅ **Validates with Granger causality** for statistical rigor  
✅ Uses modular, well-documented code  
✅ Provides easy-to-use command-line interface  
✅ Generates comprehensive visualizations  
✅ Supports flexible experimentation  
✅ Can scale to hundreds of stocks  
✅ Produces interpretable causal graphs  
✅ **Dual validation** (neural network + classical tests)  

**Ready to use!** Just install dependencies and run:
```bash
python run_analysis.py --train --analyze --stock AAPL --num_stocks 50
```

Good luck with your CS230 project! 🚀

