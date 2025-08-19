
## Project Overview

GA_tree is a genetic algorithm-based trading bot that uses tree structures to evolve trading strategies. The system combines evolutionary algorithms with GPU-accelerated prediction to create and optimize trading decision trees for cryptocurrency markets.

## Development Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Build CUDA Extension (Required)
```bash
python setup.py build_ext --inplace
```
**CRITICAL**: The project requires a C++/CUDA extension to be compiled before running. This creates the `gatree_cuda` module that provides GPU-accelerated prediction capabilities. The main script will exit with an error if this is not built first.

### Run Main Training Pipeline
```bash
python main.py
```
Executes the complete training pipeline including data download, feature generation, GA population initialization, evolution, and testing. All configuration is controlled via `experiment_config.yaml`.

### Run Individual Tests
```bash
python test_code/test_predict.py  # Test prediction functionality
python test_code/test_mu.py       # Test mutation operations
```

### Data Preprocessing Only
```bash
python preprocess_data.py
```
Run feature generation pipeline without full training.

## Architecture Overview

### Core Components

**GATree** (`models/model.py`)
- Individual genetic algorithm tree representing a trading strategy
- Uses tensor-based storage for GPU compatibility
- Implements decision nodes (feature comparisons) and action nodes (trading actions)
- Three root branches: LONG, HOLD, SHORT positions
- BFS traversal for prediction

**GATreePop** (`models/model.py`)
- Population manager for collections of GATree instances
- Supports multiprocessing for population creation
- Manages shared memory tensors for efficient GPU operations

**Evolution Engine** (`evolution/evolution.py`)
- Main controller for genetic algorithm operations
- Integrates selection, crossover, and mutation operators
- Supports elitism and mating pool selection

### GA Operators

**Selection** (`evolution/Selection/`)
- Tournament and Roulette selection strategies
- Elite selection for preserving best individuals

**Crossover** (`evolution/Crossover/`)
- Subtree, Node, RootBranch, and Chain crossover operations
- Context-aware crossover to maintain tree validity

**Mutation** (`evolution/Mutation/`)
- Add/Delete nodes and subtrees
- Parameter mutation for fine-tuning
- Node reinitialization

### Trading System

**TradingEnvironment** (`training/trading_env.py`)
- Simulates futures trading with fees, slippage, and liquidation
- Tracks positions, profit/loss, and performance metrics
- Supports funding rate calculations

**Data Pipeline** (`data/`)
- Downloads historical price data from Binance API
- Generates technical analysis features using TA-Lib
- Multi-timeframe feature engineering

### Configuration System

**experiment_config.yaml**
- Central configuration for all hyperparameters
- GA settings (population size, generations, operators)
- Trading simulation parameters  
- Feature classification rules (feature_num, feature_comparison, feature_bool)
- Data paths and training/test split ratios
- Environment settings (device, seed, output directories)

**data/feature_config.yaml**
- Technical indicator configurations
- Multi-timeframe feature generation rules

### Multiprocessing Architecture
The system uses multiprocessing for:
- GATree population initialization (`_create_tree_worker`)
- Tree reorganization after mutations (`_reorganize_worker`)
- Number of processes automatically determined by `os.cpu_count()`

## Key Implementation Details

### GPU Acceleration & Device Management
The system uses a hybrid approach:
- Python for high-level logic and data management
- C++/CUDA for performance-critical tree evaluation
- Shared memory tensors for efficient GPU-CPU communication
- Device setting controlled via `experiment_config.yaml` (e.g., 'cuda:0', 'cpu')
- Automatic fallback and error handling if CUDA not available

### Tree Structure
Trees use a tensor representation where each node has:
- Node type (ROOT_BRANCH, DECISION, ACTION)
- Parent index and depth
- Parameters defining the node's function
- Decision nodes compare features using operators (>=, <=, ==)
- Action nodes specify trading operations and parameters
- Child nodes cannot contain both action nodes and decision nodes simultaneously
- Leaf nodes must always be action nodes. A parent node that has a leaf node as its child must have exactly one child node (i.e., it should contain only one action node → to ensure decision finality).
- Therefore, all intermediate nodes are decision nodes, and all leaf nodes are action nodes.

### Feature System
Three types of feature comparisons:
1. **Feature vs Number**: RSI >= 70
2. **Feature vs Feature**: SMA_5 >= SMA_20  
3. **Feature vs Boolean**: IsBullishMarket == True

### Action Types
Trading actions include:
- NEW_LONG/NEW_SHORT: Open positions with size and leverage
- CLOSE_ALL/CLOSE_PARTIAL: Close positions entirely or partially
- ADD_POSITION: Increase existing position size
- FLIP_POSITION: Close current and open opposite position

### Evolution Strategy & Checkpointing
- Fitness based on multiple metrics: mean return, profit factor, win rate, max drawdown, compound value
- Configurable fitness weights in `experiment_config.yaml`
- Elite preservation with configurable elite size
- Warming period before elite accumulation begins  
- Automatic checkpoint saving to `results/exp_*/checkpoints/` during training
- Best population saved as `best_population.pth` for evaluation
- Resumable experiments via checkpoint loading

## File Structure Notes

- `csrc/`: C++/CUDA source files for GPU acceleration
- `models/constants.py`: Shared constants and enumerations
- `gatree_cuda.cpython-*.so`: Compiled CUDA extension (generated by setup.py)
- `test.ipynb` / `check_model.ipynb`: Jupyter notebooks for interactive testing
- Population and single tree serialization supported via PyTorch tensors
- Configuration files use UTF-8 encoding with Korean comments

## Data Pipeline

### Input Requirements
- Historical price data from Binance (OHLCV + volume)
- Optional funding rate data for futures trading simulation
- Data stored in `data_dir` path specified in config

### Output Structure
- Final processed data: `final_test_features.csv`
- Model configuration: `model_test_config.yaml`
- Results saved to `output_dir` (default: `results/exp_default/`)
- Checkpoints and best population preserved for analysis

## Dependencies

Key requirements include:
- PyTorch for tensor operations and serialization
- NumPy/Pandas for data processing  
- TA-Lib for technical analysis features
- python-binance for market data
- NetworkX/Pyvis for tree visualization
- Custom C++/CUDA extension for GPU prediction
