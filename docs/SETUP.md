# Setup Instructions

## Virtual Environment Created! ✓

A Python virtual environment has been created in the `venv/` directory.

## Activation Instructions

### On macOS/Linux:
```bash
source venv/bin/activate
```

### On Windows:
```bash
venv\Scripts\activate
```

You'll know it's activated when you see `(venv)` at the start of your terminal prompt.

## Install Dependencies

Once activated, install all required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- NumPy (numerical computing)
- Pandas (data manipulation)
- Matplotlib & Seaborn (visualization)
- scikit-learn (metrics)
- statsmodels (Granger causality tests)
- tqdm (progress bars)

## Verify Installation

Run the demo script to verify everything is set up correctly:

```bash
python demo.py
```

## Quick Start

After setup, you're ready to go:

```bash
# List available stocks
python run_analysis.py --list --num_stocks 50

# Train and analyze (quick test with 20 stocks, 5 epochs)
python run_analysis.py --train --analyze --stock AAPL --num_stocks 20 --epochs 5
```

## Deactivation

When you're done working, deactivate the virtual environment:

```bash
deactivate
```

## Full Workflow Example

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Install dependencies (first time only)
pip install -r requirements.txt

# 3. Verify setup
python demo.py

# 4. Run analysis
python run_analysis.py --train --analyze --stock AAPL --num_stocks 30 --epochs 10

# 5. When done
deactivate
```

## Troubleshooting

### "python3: command not found"
Try `python` instead of `python3`

### "pip: command not found"
After activating venv, try:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### PyTorch installation issues
If torch installation fails, visit: https://pytorch.org/get-started/locally/
Select your OS and preferences for the correct installation command.

### CUDA/GPU support
If you have an NVIDIA GPU and want to use it:
1. Check if CUDA is available: https://pytorch.org/get-started/locally/
2. Install the appropriate PyTorch version with CUDA support

The code will automatically use GPU if available, otherwise it falls back to CPU.

## IDE Integration

### VS Code
VS Code should automatically detect the virtual environment.
- Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
- Type "Python: Select Interpreter"
- Choose the one in `venv/bin/python`

### PyCharm
- Go to Preferences → Project → Python Interpreter
- Click the gear icon → Add
- Select "Existing environment"
- Choose `venv/bin/python`

### Jupyter Notebook
If you want to use the notebook:
```bash
source venv/bin/activate
pip install jupyter ipykernel
python -m ipykernel install --user --name=cs230_venv
jupyter notebook exploration.ipynb
```

## Requirements

The following packages will be installed:

```
torch>=2.0.0          # Deep learning framework
numpy>=1.24.0         # Numerical computing
pandas>=2.0.0         # Data manipulation
matplotlib>=3.7.0     # Plotting
seaborn>=0.12.0       # Statistical visualization
scikit-learn>=1.3.0   # Machine learning utilities
tqdm>=4.65.0          # Progress bars
statsmodels>=0.14.0   # Granger causality tests
```

## Next Steps

Once setup is complete, see:
- **QUICKSTART.md** - Step-by-step usage guide
- **README.md** - Full documentation
- **PROJECT_SUMMARY.md** - Overview of the framework

Happy coding! 🚀

