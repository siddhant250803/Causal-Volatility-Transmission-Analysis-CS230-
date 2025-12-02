# 🚀 Quick Reference Card

## Essential Commands

### 🧪 Testing (Start Here!)

```bash
# Quick test with enhanced visuals (5 stocks, ~15 min)
python scripts/enhanced_quick_test.py --epochs 15

# With hyperparameter tuning (~45 min)
python scripts/enhanced_quick_test.py --hp_tuning --hp_trials 10

# Test HP tuning functionality (~30 min)
python scripts/test_hyperparameter_tuning.py --n_trials 10
```

### 🎨 Basic Usage

```bash
# List available stocks
python scripts/run_analysis.py --list

# Train single stock
python scripts/run_analysis.py --train --stock AAPL

# Analyze causality
python scripts/run_analysis.py --analyze --stock AAPL
```

### ⚡ Parallel Training (AWS)

```bash
# Train all stocks in parallel
python scripts/parallel_train_all_stocks.py \
    --num_stocks 300 \
    --epochs 30 \
    --max_workers 27
```

## File Locations

### Scripts
- `scripts/enhanced_quick_test.py` - Comprehensive test with all features
- `scripts/test_hyperparameter_tuning.py` - HP tuning validation
- `scripts/parallel_train_all_stocks.py` - Multi-core training
- `scripts/run_analysis.py` - Original interface (still works!)

### Source Code
- `src/utils/enhanced_visualizations.py` - Publication-quality plots
- `src/utils/hyperparameter_tuning.py` - Automated HP search
- `src/config.py` - Configuration parameters

### Documentation
- `docs/ENHANCED_FEATURES.md` - Complete feature guide
- `IMPROVEMENTS_SUMMARY.md` - What changed
- `README.md` - Main documentation

### Output Directories
- `plots/` - All visualizations
- `checkpoints/` - Trained models
- `results/` - CSV analysis results
- `hyperparameter_search/` - HP tuning results

## Common Workflows

### 1. Quick Validation
```bash
python scripts/enhanced_quick_test.py --epochs 10 --no_granger
# Fast test, ~10 minutes
```

### 2. Full Local Test
```bash
python scripts/enhanced_quick_test.py --epochs 20 --hp_tuning --hp_trials 15
# Complete test with HP tuning, ~45-60 minutes
```

### 3. Production on AWS
```bash
# 1. Setup
pip install -r requirements.txt

# 2. Test locally first
python scripts/enhanced_quick_test.py --num_stocks 5

# 3. Scale to all stocks
python scripts/parallel_train_all_stocks.py --max_workers 27
```

## Key Parameters

### Epochs
- Quick test: `--epochs 10`
- Standard: `--epochs 20`
- Production: `--epochs 30-50`

### HP Tuning Trials
- Quick: `--hp_trials 5`
- Standard: `--hp_trials 10-15`
- Thorough: `--hp_trials 20-30`

### Parallel Workers
- Local (8 cores): `--max_workers 6`
- AWS c5.9xlarge: `--max_workers 27`
- AWS c5.18xlarge: `--max_workers 54`

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch_size 16

# Reduce workers
--max_workers 4
```

### Slow Training
```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Reduce epochs for testing
--epochs 10
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check networkx
pip install networkx>=3.0.0
```

## Expected Runtimes

| Task | Local (8 cores) | AWS (36 cores) |
|------|-----------------|----------------|
| 5 stocks (basic) | 15-20 min | N/A |
| 5 stocks + HP | 45-60 min | N/A |
| 50 stocks (parallel) | 90 min | 30 min |
| 300 stocks (parallel) | 8 hours | 2-4 hours |

## Output Examples

### Enhanced Network Graph
- Hierarchical layout with target at center
- Color-coded by influence type
- Lag information on edges
- Statistics overlay

### Strength-Lag Analysis
- 5 panels: scatter, histograms, quartiles, stats
- Top relationships annotated
- Mean/median markers

### Hyperparameter Results
- JSON file with all trials
- Visualization showing parameter effects
- Best configuration extracted

## Tips

✅ Start with `enhanced_quick_test.py` (5 stocks)  
✅ Enable HP tuning for 1-2 stocks first  
✅ Use best config for remaining stocks  
✅ Check `plots/` after each run  
✅ Monitor AWS costs when using large instances  

## Help & Documentation

```bash
# Get help for any script
python scripts/enhanced_quick_test.py --help
python scripts/parallel_train_all_stocks.py --help
python scripts/test_hyperparameter_tuning.py --help
```

For detailed documentation, see:
- `docs/ENHANCED_FEATURES.md`
- `IMPROVEMENTS_SUMMARY.md`
- `README.md`

---

**Quick start:** `python scripts/enhanced_quick_test.py --epochs 10`


