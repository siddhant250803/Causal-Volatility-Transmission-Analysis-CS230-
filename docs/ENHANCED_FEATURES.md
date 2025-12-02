# Enhanced Features Guide

This document describes the new enhanced features added to the Causal Volatility Transmission framework.

## 🎨 Enhanced Visualizations

### Overview
Publication-quality visualizations with improved aesthetics, better information density, and interactive network graphs.

### Features

#### 1. Enhanced Network Graph
- **Hierarchical layouts** with target stock at center
- **Curved arrows** showing causal direction
- **Color-coded nodes** by role (target, primary sources, secondary sources)
- **Size-based influence** (node size proportional to connections)
- **Lag-colored edges** (edge color indicates time lag)
- **Statistics overlay** showing network metrics

```python
from src.utils import EnhancedVisualizer

visualizer = EnhancedVisualizer()
visualizer.plot_network_graph(
    relationships_df,
    stock_name="AAPL",
    threshold=0.1,
    layout='hierarchical'  # or 'spring', 'circular'
)
```

#### 2. Comprehensive Strength-Lag Analysis
- **Multi-panel layout** with 5 coordinated views
- **Scatter plot** with annotated top relationships
- **Distributions** for both strength and lag
- **Quartile analysis** showing strength patterns by lag
- **Statistics table** with key metrics

```python
visualizer.plot_strength_lag_analysis(relationships_df, "AAPL")
```

#### 3. Enhanced Heatmap Matrix
- **Full causal matrix** for all stock pairs
- **Masked diagonal** (no self-causation)
- **Annotated values** for small matrices
- **Colorbar** with clear scale
- **Asymmetric relationships** clearly visible

```python
visualizer.plot_heatmap_matrix(causal_matrix, title="Full Network")
```

#### 4. Training History Plots
- **Multi-metric tracking** (loss, RMSE, R², learning rate)
- **Train/validation comparison**
- **Best epoch marker**
- **Learning rate schedule** (if using scheduler)

```python
visualizer.plot_training_history(history, "AAPL")
```

## 🔬 Hyperparameter Tuning

### Overview
Randomized search over hyperparameter space with early stopping and automatic result tracking.

### Features

- **Flexible search space** definition (uniform, log-uniform, choice)
- **Early stopping** per trial (efficiency)
- **Automatic result saving** (JSON format)
- **Visualization** of search results
- **Best configuration** extraction

### Usage

```python
from src.utils import HyperparameterSearch, get_default_search_space

# Define search space
search_space = get_default_search_space()
# Or custom:
search_space = {
    'LEARNING_RATE': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-2},
    'D_MODEL': {'type': 'choice', 'values': [64, 128, 256]},
    'DROPOUT': {'type': 'uniform', 'low': 0.0, 'high': 0.3}
}

# Run search
hp_search = HyperparameterSearch(config, search_space, n_trials=20)
best_config, best_result = hp_search.run_search(
    train_loader, val_loader,
    n_stocks, target_stock_idx, "AAPL",
    max_epochs_per_trial=30
)

# Plot results
hp_search.plot_search_results("AAPL")
```

### Default Search Space

| Parameter | Type | Range |
|-----------|------|-------|
| LEARNING_RATE | log_uniform | [1e-4, 1e-2] |
| D_MODEL | choice | [32, 64, 128, 256] |
| D_K | choice | [16, 32, 64] |
| DROPOUT | uniform | [0.0, 0.3] |
| LAMBDA_GATE | log_uniform | [1e-5, 1e-3] |
| GAMMA_TV | log_uniform | [1e-5, 1e-3] |
| BATCH_SIZE | choice | [16, 32, 64] |

### Output

Results are saved to `hyperparameter_search/` directory:
- `{stock}_{timestamp}.json` - Complete search results
- `{stock}_search_plot.png` - Visualization of parameter effects

## ⚡ Parallel Training

### Overview
Multi-core parallel training for processing all stocks simultaneously. Optimized for AWS EC2 instances.

### Features

- **Process-based parallelism** using multiprocessing
- **Automatic worker allocation** (75% of CPU cores)
- **Progress tracking** with tqdm
- **Error handling** per stock
- **Result aggregation** with performance summary

### Usage

```bash
# Train all stocks in parallel
python scripts/parallel_train_all_stocks.py

# With custom settings
python scripts/parallel_train_all_stocks.py \
    --num_stocks 50 \
    --epochs 30 \
    --max_workers 8 \
    --batch_size 64
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_stocks` | All | Number of stocks to train |
| `--epochs` | Config | Training epochs per stock |
| `--max_workers` | Auto (75%) | Parallel workers |
| `--batch_size` | Config | Batch size |

### Output

- Training completes in parallel across workers
- Results saved to `results/parallel_training_{timestamp}.csv`
- Individual models saved to `checkpoints/`

### AWS Optimization

For AWS EC2 instances:

```bash
# Recommended instance types:
# - c5.9xlarge (36 vCPUs) -> 27 workers
# - c5.18xlarge (72 vCPUs) -> 54 workers

# Launch with maximum parallelism
python scripts/parallel_train_all_stocks.py \
    --num_stocks 300 \
    --max_workers 50 \
    --batch_size 64
```

## 🧪 Testing Scripts

### 1. Hyperparameter Tuning Test

Test HP tuning with 5 stocks:

```bash
python scripts/test_hyperparameter_tuning.py

# Quick validation
python scripts/test_hyperparameter_tuning.py --n_trials 5 --max_epochs 10

# Thorough search
python scripts/test_hyperparameter_tuning.py --n_trials 20 --max_epochs 30
```

### 2. Enhanced Quick Test

Comprehensive test with all new features:

```bash
# Basic test with enhanced visuals
python scripts/enhanced_quick_test.py

# With hyperparameter tuning
python scripts/enhanced_quick_test.py --hp_tuning --hp_trials 15

# Full featured test
python scripts/enhanced_quick_test.py \
    --epochs 30 \
    --hp_tuning \
    --hp_trials 20
```

## 📊 Output Structure

```
project/
├── hyperparameter_search/      # HP tuning results
│   ├── {stock}_{timestamp}.json
│   └── {stock}_search_plot.png
│
├── plots/                      # Enhanced visualizations
│   ├── {stock}_enhanced_network.png
│   ├── {stock}_strength_lag_analysis.png
│   ├── {stock}_training_history.png
│   └── causal_matrix_heatmap.png
│
├── results/                    # Analysis results
│   ├── enhanced_quick_test_relationships.csv
│   ├── enhanced_quick_test_matrix.csv
│   ├── enhanced_quick_test_summary.csv
│   └── parallel_training_{timestamp}.csv
│
└── checkpoints/                # Trained models
    ├── {stock}_best.pt
    └── {stock}_stock_info.json
```

## 🚀 Complete Workflow Example

### 1. Test with 5 Stocks + HP Tuning

```bash
# Run comprehensive test with HP tuning
python scripts/enhanced_quick_test.py \
    --num_stocks 5 \
    --epochs 20 \
    --hp_tuning \
    --hp_trials 10
```

**Expected time:** ~30-60 minutes (depending on hardware)

**Outputs:**
- 5 trained models (with optimized hyperparameters)
- Enhanced visualizations for each stock
- HP tuning results and plots
- Combined causal matrix

### 2. Scale to All Stocks

```bash
# Use best HP config and train all stocks in parallel
python scripts/parallel_train_all_stocks.py \
    --epochs 30 \
    --max_workers 8
```

**Expected time:** Depends on # stocks and workers
- 50 stocks, 8 workers: ~2-3 hours
- 300 stocks, 50 workers (AWS): ~2-4 hours

### 3. Generate Final Visualizations

```bash
# After training, create comprehensive visualizations
python scripts/create_full_network_viz.py --threshold 0.15
```

## 🎯 Best Practices

### For Local Development
1. Test with `enhanced_quick_test.py` (5 stocks)
2. Enable HP tuning for 1-2 stocks to find good parameters
3. Use those parameters for remaining stocks

### For AWS Training
1. Use `c5` or `c6i` instance families (compute-optimized)
2. Start with smaller subset to validate
3. Monitor memory usage (adjust batch size if needed)
4. Use `parallel_train_all_stocks.py` for full dataset

### For Hyperparameter Tuning
1. Start with 10-15 trials for initial exploration
2. Narrow search space based on initial results
3. Run 20-30 trials for final optimization
4. Use `max_epochs_per_trial=20-30` for good balance

## 🔧 Configuration Tips

### For Better Visualizations
```python
# In src/config.py
PLOT_DIR = "plots/"
SAVE_PLOTS = True

# Increase top K for more relationships in plots
TOP_K_INFLUENCES = 20

# Adjust threshold for network clarity
CAUSAL_THRESHOLD = 0.15  # Higher = sparser networks
```

### For Faster HP Tuning
```python
# Reduce search space dimensionality
search_space = {
    'LEARNING_RATE': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-2},
    'D_MODEL': {'type': 'choice', 'values': [64, 128]},  # Fewer choices
    'DROPOUT': {'type': 'uniform', 'low': 0.1, 'high': 0.2}  # Narrower range
}

# Use fewer trials with early stopping
n_trials = 10
max_epochs_per_trial = 15
```

## 📈 Performance Improvements

Compared to baseline:

| Feature | Speedup | Notes |
|---------|---------|-------|
| Parallel training (8 cores) | ~6-7x | Near-linear scaling |
| HP tuning with early stop | ~2x | vs full epochs all trials |
| Enhanced viz (batch) | ~1.5x | Multiple plots at once |

## 🐛 Troubleshooting

### Memory Issues
```bash
# Reduce batch size
--batch_size 16

# Reduce workers
--max_workers 4

# Use smaller models in HP search
search_space['D_MODEL'] = {'type': 'choice', 'values': [32, 64]}
```

### Slow Training
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Reduce max_lag (faster attention)
# In src/config.py:
MAX_LAG = 8  # instead of 12
```

### Visualization Errors
```bash
# Install required packages
pip install networkx>=3.0.0 matplotlib>=3.7.0

# Check plot directory exists
mkdir -p plots
```

## 📚 API Reference

See individual module docstrings for detailed API documentation:
- `src/utils/enhanced_visualizations.py`
- `src/utils/hyperparameter_tuning.py`
- `scripts/parallel_train_all_stocks.py`

---

**For questions or issues, please refer to the main README.md or open an issue on GitHub.**


