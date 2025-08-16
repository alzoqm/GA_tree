# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GA_tree is a genetic algorithm-based trading bot that uses tree structures to evolve trading strategies. The system combines evolutionary algorithms with GPU-accelerated prediction to create and optimize trading decision trees for cryptocurrency markets.

## Development Commands

### Build CUDA Extension
```bash
python setup.py build_ext --inplace
```
The project requires a C++/CUDA extension to be compiled before running. This creates the `gatree_cuda` module that provides GPU-accelerated prediction capabilities.

### Run Main Training Pipeline
```bash
python main.py
```
Executes the complete training pipeline including data download, feature generation, GA population initialization, evolution, and testing.

### Run Tests
```bash
python test_code/test_predict.py
python test_code/test_mu.py
```
Test prediction functionality and mutation operations.

### Install Dependencies
```bash
pip install -r requirements.txt
```

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

### Configuration

**experiment_config.yaml**
- Central configuration for all hyperparameters
- GA settings (population size, generations, operators)
- Trading simulation parameters
- Feature classification rules

**data/feature_config.yaml**
- Technical indicator configurations
- Multi-timeframe feature generation rules

## Key Implementation Details

### GPU Acceleration
The system uses a hybrid approach:
- Python for high-level logic and data management
- C++/CUDA for performance-critical tree evaluation
- Shared memory tensors for efficient GPU-CPU communication

### Tree Structure
Trees use a tensor representation where each node has:
- Node type (ROOT_BRANCH, DECISION, ACTION)
- Parent index and depth
- Parameters defining the node's function
- Decision nodes compare features using operators (>=, <=, ==)
- Action nodes specify trading operations and parameters

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

### Evolution Strategy
- Fitness based on multiple metrics: mean return, profit factor, win rate, max drawdown, compound value
- Elite preservation with configurable elite size
- Warming period before elite accumulation begins
- Checkpoint saving for experiment resumption

## File Structure Notes

- `csrc/`: C++/CUDA source files for GPU acceleration
- `models/constants.py`: Shared constants and enumerations
- `verify_predictions.py`: Standalone prediction verification script
- Test files demonstrate tree construction and prediction validation
- Population and single tree serialization supported via PyTorch tensors

## Dependencies

Key requirements include:
- PyTorch for tensor operations and serialization
- NumPy/Pandas for data processing  
- TA-Lib for technical analysis features
- python-binance for market data
- NetworkX/Pyvis for tree visualization
- Custom C++/CUDA extension for GPU prediction