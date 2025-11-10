# Causal Volatility Transmission Analysis

## Overview

This framework implements an **attention-based causal inference model** to map intraday volatility transmission across stocks using high-frequency (5-minute) trading data. The model learns:

1. **Which stocks influence others** (causal direction)
2. **How strong the influence is** (causal strength via learned gates)
3. **The time delay of influence** (learned lags in 5-minute intervals)

Based on the CS230 project: *"Mapping Intraday Volatility Transmission Across 300 Stocks Using Attention-Based Causal Inference"*

## Project Structure

```
.
├── config.py                 # Configuration parameters
├── data/
│   ├── __init__.py
│   └── dataloader.py        # Data loading and preprocessing
├── models/
│   ├── __init__.py
│   └── attention_model.py   # Attention-based causal model
├── utils/
│   ├── __init__.py
│   ├── losses.py           # Loss functions with regularization
│   └── metrics.py          # Evaluation metrics
├── train.py                # Training script
├── analyze_causality.py    # Causal analysis and visualization
├── run_analysis.py         # Main interactive script (USE THIS!)
├── checkpoints/            # Saved models
├── plots/                  # Generated visualizations
└── results/                # Analysis results (CSV)
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the data file `HF_Returns_Stocks.csv` in the project root.

## Quick Start

### 1. List Available Stocks
```bash
python run_analysis.py --list
```

### 2. Train Model for a Stock
Train a model to predict volatility for AAPL using information from other stocks:
```bash
python run_analysis.py --train --stock AAPL
```

### 3. Analyze Causal Relationships
After training, analyze which stocks causally influence AAPL:
```bash
python run_analysis.py --analyze --stock AAPL
```

### 4. Train and Analyze (One Command)
```bash
python run_analysis.py --train --analyze --stock AAPL
```

## Advanced Usage

### Custom Parameters

```bash
# Train with more stocks and custom epochs
python run_analysis.py --train --stock NVDA --num_stocks 100 --epochs 30

# Analyze with custom threshold and top K
python run_analysis.py --analyze --stock NVDA --top_k 15 --threshold 0.15
```

### Configuration

Edit `config.py` to modify:
- Model architecture (embedding dimensions, attention heads, etc.)
- Training parameters (learning rate, batch size, epochs)
- Regularization weights (gate sparsity, temporal smoothness)
- Data processing (lookback window, prediction horizon)

## Model Architecture

### Key Components

1. **Learned Lag Attention Mechanism**
   - Each stock has a learned lag parameter
   - Determines optimal time delay for information transmission
   - Range: 0 to 12 intervals (0 to 60 minutes)

2. **Causal Gates**
   - Learnable parameters for each stock
   - Control which stocks have causal influence
   - Sparse regularization promotes interpretability

3. **Prediction Network**
   - Combines target stock's own history with cross-stock influences
   - Multi-layer perceptron predicts next-interval volatility

### Loss Function

```
L = MSE + λ·||g||_{2,1} + γ·TV(α) + η·IRM
```

Where:
- **MSE**: Prediction accuracy
- **||g||_{2,1}**: Group lasso on causal gates (sparsity)
- **TV(α)**: Total variation on attention weights (smoothness)
- **IRM**: Invariant risk minimization (regime stability)

## Output

### 1. Trained Models
Saved to `checkpoints/{STOCK}_best.pt`

### 2. Analysis Report
Terminal output showing:
- Top influencing stocks
- Causal strengths
- Lag statistics

Example:
```
TOP 10 INFLUENCING STOCKS:
source_stock  target_stock  causal_strength  lag_minutes  lag_intervals
         SPY          AAPL           0.8945         25.0              5
        MSFT          AAPL           0.7823         15.0              3
        NVDA          AAPL           0.7156         30.0              6
         ...           ...              ...          ...            ...
```

### 3. CSV Results
Saved to `results/{STOCK}_causal_relationships.csv`

### 4. Visualizations
Saved to `plots/`:

- **Causal Network**: Bar chart showing top influences with lag information
- **Lag Distribution**: Histogram and scatter plot of lags vs. strengths
- **Heatmap**: Visual representation of causal influence matrix

## Example Workflow

```bash
# 1. Explore available stocks
python run_analysis.py --list

# 2. Train models for multiple stocks
python run_analysis.py --train --stock AAPL
python run_analysis.py --train --stock NVDA
python run_analysis.py --train --stock MSFT

# 3. Analyze causal relationships
python run_analysis.py --analyze --stock AAPL --top_k 15
python run_analysis.py --analyze --stock NVDA --top_k 15
python run_analysis.py --analyze --stock MSFT --top_k 15

# 4. Compare results
# Check results/ and plots/ directories
```

## Key Features

✅ **Modular architecture** - Easy to extend and customize  
✅ **Learned time lags** - Discovers temporal dependencies automatically  
✅ **Causal interpretability** - Clear influence paths with strengths  
✅ **Regularized learning** - Sparse, stable, and smooth causal graphs  
✅ **Interactive interface** - Simple command-line usage  
✅ **Comprehensive visualization** - Multiple plot types for analysis  
✅ **Scalable** - Handles hundreds of stocks efficiently  

## Technical Details

### Data Processing
- **Input**: 5-minute high-frequency returns
- **Preprocessing**: Z-score normalization per stock
- **Target**: Realized volatility (rolling 1-hour window)
- **Splits**: 70% train, 15% validation, 15% test (chronological)

### Model Parameters
- Embedding dimension: 64
- Query/Key/Value dimension: 32
- Max lag: 12 intervals (60 minutes)
- Lookback window: 12 intervals (60 minutes)
- Prediction horizon: 1 interval (5 minutes)

### Training
- Optimizer: Adam (lr=0.001)
- Batch size: 32
- Early stopping: Patience of 10 epochs
- Gradient clipping: max norm = 1.0

## Limitations & Future Work

- Current implementation trains one model per target stock
- For full network, train models for all stocks
- Could add sector/industry information as features
- Could implement regime detection for time-varying analysis
- Could add more sophisticated IRM implementation

## Citation

If you use this framework, please cite:

```
Sukhani, S., Roger, A., & Alhusseini, M. (2025). 
Mapping Intraday Volatility Transmission Across 300 Stocks 
Using Attention-Based Causal Inference. 
CS230 Deep Learning Project, Stanford University.
```

Data source:
```
Pelger, M. (2020). Understanding Systematic Risk: A High-Frequency Approach.
```

## License

MIT License - See repository for details

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

