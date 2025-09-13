

# GA-Tree Trading Bot

This repository contains a high-performance framework for discovering cryptocurrency futures trading strategies using a massively parallel Genetic Algorithm (GA). The strategies are evolved as decision trees and are rigorously evaluated using Walk-Forward Optimization (WFO) to ensure robustness and prevent overfitting.

The entire core of the GA, including population evaluation, selection, crossover, and mutation, is accelerated with custom C++/CUDA kernels, allowing for the exploration of vast search spaces with large populations and complex feature sets.

## Key Features

-   **Genetic Algorithm with Tree-based Strategies**: Evolves trading logic as interpretable decision trees (`GATree`), optimizing for complex market conditions.
-   **Massively Parallel GPU Acceleration**: Custom CUDA kernels for all critical GA operations (prediction, selection, crossover, mutation) and simulation components, enabling large-scale experiments that are orders of magnitude faster than CPU-based alternatives.
-   **Walk-Forward Optimization (WFO)**: Implements a robust backtesting framework using rolling time windows for training (In-Sample) and validation (Out-of-Sample) to build strategies that are resilient to market regime changes.
-   **Multi-Objective Fitness Function**: Optimizes for multiple performance metrics simultaneously (e.g., return, drawdown, Sharpe Ratio, PSR) using Pareto optimization, allowing for a nuanced selection of strategies based on a user-defined risk/reward profile.
-   **Comprehensive Trading Simulator**: A realistic futures trading environment that models:
    -   Dynamic, volatility-based slippage.
    -   Funding rates.
    -   Leverage and deterministic liquidation.
    -   Taker fees and insurance fund fees.
-   **Advanced Feature Engineering Pipeline**: A data processing pipeline that generates multi-timeframe features using TA-Lib and custom composite indicators, all managed via a central YAML configuration.
-   **Checkpointing and Resumption**: A robust mechanism to save and resume long-running training sessions automatically or on-demand.
-   **Integrated Analysis & Logging**: Includes built-in tools for analyzing the best-evolved models and logging experiment metrics, parameters, and artifacts to Weights & Biases.

## Architecture Overview

The project is structured into several key components:

-   **/csrc**: Contains the C++/CUDA source code for all performance-critical operations. These are compiled into Python extensions via `setup.py`.
-   **/gatree\_cuda\_compat**: A compatibility module that imports and exposes all compiled CUDA kernels under a single, easy-to-use namespace.
-   **/data**: Handles the data pipeline, including downloading from Binance (`data_download.py`), deterministic funding/mark price estimation (`funding_mark_proxy.py`), and feature generation (`ta_lib_feature_generator.py`, `merge_dataset.py`).
-   **/models**: Defines the core data structures for trading strategies: `GATree` (an individual strategy) and `GATreePop` (a population of strategies).
-   **/evolution**: Implements the genetic operators—Selection (`PPTSSelection`), Crossover, and Mutation—and the main `Evolution` engine that orchestrates the evolutionary process.
-   **/training**: Contains the high-level logic for running experiments, including the `TradingEnvironment`, `FitnessSimulator`, `WalkForwardOptimizer`, and the main `runner` scripts.
-   **/analysis**: Provides tools for post-run analysis (`analyze_best.py`), plotting (`plots.py`), statistical tests (`stat_tests.py`), and Weights & Biases integration (`wandb_logger.py`).
-   **/utils**: Utility modules, including the `CheckpointManager`.

## Getting Started

### Prerequisites

-   NVIDIA GPU with CUDA support
-   CUDA Toolkit (11.x or newer recommended)
-   C++ Compiler (GCC/G++, Clang, or MSVC)
-   Python 3.9+
-   [TA-Lib](https://github.com/mrjbq7/ta-lib) C library installed on your system.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file. Based on the imports, it should include `torch`, `pandas`, `numpy`, `pyyaml`, `tqdm`, `python-binance`, `python-dotenv`, `pyvis`, `networkx`, `scikit-learn`, and `wandb`.)*

4.  **Compile the CUDA extensions:**
    This step compiles the C++/CUDA source files into Python modules.
    ```bash
    python setup.py build_ext --inplace
    ```

5.  **Set up Environment Variables:**
    Create a `.env` file in the project root for API keys:
    ```env
    # .env
    BINANCE_API_KEY="YOUR_BINANCE_API_KEY"
    BINANCE_API_SECRET="YOUR_BINANCE_SECRET_KEY"
    WANDB_API_KEY="YOUR_WANDB_API_KEY"
    ```

## Usage

Running an experiment involves three main steps: configuring the experiment, preprocessing the data, and running the main training pipeline.

### Step 1: Configure the Experiment

Modify `experiment_config.yaml` to define all aspects of your experiment. This is the central control file.

-   **`environment`**: Set the CUDA device, random seed, and output directories.
-   **`data`**: Specify the trading symbol (e.g., `BTCUSDT`), interval, and feature engineering settings.
-   **`ga_core`**: Define the GA parameters like population size, number of generations, and elite size.
-   **`operators`**: Configure the probabilities and behaviors of crossover and mutation operators.
-   **`walk_forward`**: Enable and configure the WFO parameters, including window sizes and step sizes.
-   **`evaluation`**: Set up the simulation environment (fees, slippage) and define the multi-objective fitness functions for in-sample (IS), out-of-sample (OOS), and final aggregation.
-   **`feature_classification`**: Categorize your features to control how they are used within the decision tree nodes.

### Step 2: Preprocess Data

Run the preprocessing script to download market data, generate features, and save the final dataset. The script is idempotent and will skip execution if the final files already exist.

```bash
python preprocess_data.py
```

This will create `final_test_features.csv` and `model_test_config.yaml` in the directory specified by `data_dir` in your config.

### Step 3: Run the Training Pipeline

Execute the main script to start the Walk-Forward Optimization process.

```bash
# Start a new run
python main.py

# Resume a run from the latest checkpoint
python main.py --resume

# Clear all existing checkpoints and start fresh
python main.py --clear-checkpoints
```

The process will iterate through the WFO windows, performing training and evaluation as configured. Final results, including the best-evolved population, are saved in the `output_dir`.

### Step 4: Analyze the Results

After a run is complete, you can analyze the best-evolved models to understand their structure and behavior.

```bash
python analysis/analyze_best.py --config experiment_config.yaml```

This script generates several outputs in the `analysis/` sub-directory of your experiment's results folder, including:
-   `feature_usage.png`: Bar chart of the most frequently used features.
-   `comparison_type_mix.png`: Distribution of comparison types (e.g., feature-vs-number).
-   `feature_interaction.png`: A heatmap of feature co-occurrence in decision nodes.
-   `permutation_importance.png`: Feature importance based on performance degradation when a feature's values are shuffled.
-   JSON files containing the raw data for the plots.

## Testing

The project includes a suite of tests using `pytest` to verify the integrity of the CUDA kernels and core Python logic. To run the tests, ensure you have `pytest` installed and execute it from the project root:

```bash
pytest
```

**Note**: The tests marked with `@pytest.mark.cuda` require a CUDA-enabled environment and compiled extensions to run.