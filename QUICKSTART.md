# Quick Start Guide

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify setup
python demo.py
```

## Usage in 3 Steps

### Step 1: List Available Stocks

```bash
python run_analysis.py --list --num_stocks 50
```

This will show you the first 50 stock tickers available in the dataset.

### Step 2: Train a Model

Train a model to predict volatility for a specific stock (e.g., AAPL):

```bash
python run_analysis.py --train --stock AAPL --num_stocks 20 --epochs 5
```

**Parameters:**
- `--stock AAPL`: Target stock to predict
- `--num_stocks 20`: Use 20 stocks for faster training (good for testing)
- `--epochs 5`: Train for 5 epochs (increase for better results, e.g., 50)

**What happens:**
- Loads 5-minute high-frequency returns data
- Computes realized volatility
- Trains attention model with learned lags and causal gates
- Saves best model to `checkpoints/AAPL_best.pt`

### Step 3: Analyze Causal Relationships

```bash
python run_analysis.py --analyze --stock AAPL --top_k 15
```

**Parameters:**
- `--top_k 15`: Show top 15 influencing stocks
- `--threshold 0.1`: Minimum causal strength (default)

**Output:**
1. **Console Report**: Table of influencing stocks with strengths and lags
2. **CSV File**: `results/AAPL_causal_relationships.csv`
3. **Plots**: Saved to `plots/` directory
   - Causal network bar chart
   - Lag distribution histogram
   - Causal strength heatmap

## One-Command Workflow

Train and analyze in a single command:

```bash
python run_analysis.py --train --analyze --stock AAPL --num_stocks 20 --epochs 10
```

## Example Output

After running analysis, you'll see:

```
================================================================================
CAUSAL ANALYSIS REPORT: AAPL
================================================================================

Found 18 significant causal relationships
(Threshold: 0.1)

────────────────────────────────────────────────────────────────────────────────
TOP 10 INFLUENCING STOCKS:
────────────────────────────────────────────────────────────────────────────────
source_stock  target_stock  causal_strength  lag_minutes  lag_intervals
         SPY          AAPL           0.8945         25.0              5
        MSFT          AAPL           0.7823         15.0              3
        NVDA          AAPL           0.7156         30.0              6
        INTC          AAPL           0.6543         20.0              4
         IBM          AAPL           0.5892         35.0              7
        CSCO          AAPL           0.5234         10.0              2
        ORCL          AAPL           0.4987         40.0              8
        QCOM          AAPL           0.4623         25.0              5
        ADBE          AAPL           0.4321         15.0              3
        AMZN          AAPL           0.4012         30.0              6
```

## Interpretation

**Causal Strength (0-1):**
- Values closer to 1 indicate stronger influence
- Controlled by learned "causal gates" in the model

**Lag (minutes):**
- Time delay before influence is observed
- E.g., "25.0 min" means SPY movements affect AAPL after ~25 minutes
- Learned automatically by the model

## Advanced Usage

### More Stocks, Better Results

```bash
# Use 100 stocks and train for 30 epochs
python run_analysis.py --train --stock AAPL --num_stocks 100 --epochs 30
```

### Adjust Threshold

```bash
# Only show very strong relationships (>0.2)
python run_analysis.py --analyze --stock AAPL --threshold 0.2 --top_k 10
```

### Compare Multiple Stocks

```bash
# Train models for several stocks
python run_analysis.py --train --stock AAPL --num_stocks 50
python run_analysis.py --train --stock NVDA --num_stocks 50
python run_analysis.py --train --stock MSFT --num_stocks 50

# Analyze each
python run_analysis.py --analyze --stock AAPL
python run_analysis.py --analyze --stock NVDA
python run_analysis.py --analyze --stock MSFT

# Compare results in results/ directory
```

## Understanding the Model

### What the Model Learns

1. **For each stock pair (Source → Target):**
   - Does Source influence Target? (via causal gate)
   - How strong is the influence? (gate value 0-1)
   - What is the time lag? (0-60 minutes)

2. **Regularization ensures:**
   - Sparsity: Only important stocks have influence
   - Smoothness: Attention patterns are temporally coherent
   - Stability: Relationships are consistent across market regimes

### Key Features

- **Attention Mechanism**: Identifies which stocks to focus on
- **Learned Lags**: Discovers optimal time delays automatically
- **Causal Gates**: Sparse selection of influential stocks
- **Interpretable**: Clear cause-effect relationships with timing

## Troubleshooting

### "Stock not found"
- Check available stocks with `--list`
- Ensure correct spelling (case-sensitive)

### "Model checkpoint not found"
- Train the model first with `--train`

### Training too slow
- Reduce `--num_stocks` (try 20 for testing)
- Reduce `--epochs` (try 5 for quick test)
- Check if GPU is available (runs on CPU if not)

### Out of memory
- Reduce `BATCH_SIZE` in `config.py`
- Reduce `--num_stocks`

## Next Steps

1. **Customize Configuration**: Edit `config.py`
   - Model architecture (dimensions, layers)
   - Training parameters (learning rate, batch size)
   - Regularization weights

2. **Explore Code**: Check the modular structure
   - `data/dataloader.py` - Data preprocessing
   - `models/attention_model.py` - Model architecture
   - `utils/losses.py` - Loss functions
   - `analyze_causality.py` - Analysis tools

3. **Extend Framework**:
   - Add sector information
   - Implement time-varying analysis
   - Add more baseline models for comparison

## Questions?

See `README.md` for detailed documentation or check the code comments.

Happy analyzing! 🚀

