# training/multi_objective_trading_env.py

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from .trading_env import TradingEnvironment

class MultiObjectiveTradingEnvironment(TradingEnvironment):
    """
    Multi-Objective Trading Environment extending the base TradingEnvironment.
    
    This environment supports multiple optimization objectives simultaneously,
    returning objective vectors instead of single fitness scores. Supports
    configurable objective combinations and maintains compatibility with
    the existing trading simulation logic.
    """
    
    def __init__(self, chromosomes_size: int, env_config: dict, 
                 objectives_config: dict, device='cpu'):
        """
        Initialize Multi-Objective Trading Environment.
        
        Args:
            chromosomes_size (int): Number of individuals in the population
            env_config (dict): Trading environment configuration
            objectives_config (dict): Multi-objective configuration
            device (str): Device for tensor operations
        """
        super().__init__(chromosomes_size, env_config, device)
        
        # Multi-objective configuration
        self.objectives_config = objectives_config
        self.objective_names = objectives_config.get('objectives', [
            'mean_return', 'max_drawdown', 'sharpe_ratio', 'win_rate'
        ])
        self.minimize_objectives = objectives_config.get('minimize', [
            False, True, False, False  # minimize drawdown, maximize others
        ])
        
        if len(self.minimize_objectives) != len(self.objective_names):
            raise ValueError("minimize list must match objectives list length")
        
        # Additional tracking for multi-objective metrics
        self.risk_free_rate = objectives_config.get('risk_free_rate', 0.0)
        self.lookback_period = objectives_config.get('lookback_period', 252)  # for annualization
        
        # Enhanced statistics for Sharpe ratio and other risk metrics
        self.returns_history = []  # Store return history for rolling calculations
        self.equity_history = []   # Store equity history for advanced metrics
        self.volatility_lookback = objectives_config.get('volatility_lookback', 30)
        
    def step(self, market_data, next_open_price, predicted_actions):
        """
        Enhanced step function that tracks additional metrics for multi-objective optimization.
        
        Args:
            market_data: Current market data row
            next_open_price: Next candle's opening price
            predicted_actions: Predicted actions tensor
        """
        # Store equity before step for return calculation
        pre_equity = self._calculate_current_equity(market_data)
        
        # Call parent step function
        super().step(market_data, next_open_price, predicted_actions)
        
        # Store equity after step for return calculation
        post_equity = self._calculate_current_equity(market_data)
        
        # Calculate and store period returns
        returns = (post_equity - pre_equity) / (pre_equity + 1e-9)
        self.returns_history.append(returns)
        self.equity_history.append(post_equity)
        
        # Keep only recent history for memory efficiency
        max_history = max(self.volatility_lookback, self.lookback_period)
        if len(self.returns_history) > max_history:
            self.returns_history.pop(0)
            self.equity_history.pop(0)
    
    def _calculate_current_equity(self, market_data) -> torch.Tensor:
        """
        Calculate current equity including unrealized P&L.
        
        Args:
            market_data: Current market data for mark-to-market calculation
            
        Returns:
            equity: [chromosomes_size] tensor of current equity values
        """
        # Use realized cumulative + unrealized P&L
        curr_high = torch.tensor(market_data['High'], dtype=torch.float32, device=self.device)
        curr_low = torch.tensor(market_data['Low'], dtype=torch.float32, device=self.device)
        curr_close = torch.tensor(market_data['Close'], dtype=torch.float32, device=self.device)
        mark_t = (curr_high + curr_low + curr_close) / 3.0
        
        # Calculate unrealized P&L for open positions
        open_mask = self.pos_list != 0
        unrealized_pnl = torch.zeros_like(self.realized_cum)
        
        if open_mask.any():
            side = torch.where(self.pos_list == 2, 1.0, -1.0)  # long=+1, short=-1
            entry = self.price_list
            lev = self.leverage_ratio.float()
            er = self.enter_ratio
            pnl_pct = side * (mark_t - entry) / (entry + 1e-9)
            unrealized_pnl[open_mask] = (pnl_pct * lev * er)[open_mask]
        
        # Total equity = realized + unrealized, starting from 1.0
        equity = 1.0 + self.realized_cum + unrealized_pnl
        return equity
    
    def _calculate_sharpe_ratio(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Calculate Sharpe ratio for each individual.
        
        Args:
            returns: [time_steps, chromosomes_size] tensor of period returns
            
        Returns:
            sharpe_ratios: [chromosomes_size] tensor of Sharpe ratios
        """
        if len(returns) < 2:
            return torch.zeros(self.chromosomes_size, device=self.device)
        
        returns_tensor = torch.stack(returns)  # [time_steps, chromosomes_size]
        
        # Calculate excess returns (assuming risk-free rate is annualized)
        period_rf_rate = self.risk_free_rate / self.lookback_period
        excess_returns = returns_tensor - period_rf_rate
        
        # Calculate mean and std of excess returns
        mean_excess = excess_returns.mean(dim=0)  # [chromosomes_size]
        std_excess = excess_returns.std(dim=0, unbiased=True)  # [chromosomes_size]
        
        # Sharpe ratio with numerical stability
        sharpe_ratios = torch.where(
            std_excess > 1e-8,
            mean_excess / std_excess * np.sqrt(self.lookback_period),  # Annualized
            torch.zeros_like(mean_excess)
        )
        
        return sharpe_ratios
    
    def _calculate_volatility(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Calculate volatility for each individual.
        
        Args:
            returns: [time_steps, chromosomes_size] tensor of period returns
            
        Returns:
            volatilities: [chromosomes_size] tensor of annualized volatilities
        """
        if len(returns) < 2:
            return torch.zeros(self.chromosomes_size, device=self.device)
        
        returns_tensor = torch.stack(returns)  # [time_steps, chromosomes_size]
        volatilities = returns_tensor.std(dim=0, unbiased=True) * np.sqrt(self.lookback_period)
        
        return volatilities
    
    def _calculate_calmar_ratio(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Calculate Calmar ratio (annual return / max drawdown).
        
        Args:
            returns: [time_steps, chromosomes_size] tensor of period returns
            
        Returns:
            calmar_ratios: [chromosomes_size] tensor of Calmar ratios
        """
        if len(returns) < 2:
            return torch.zeros(self.chromosomes_size, device=self.device)
        
        returns_tensor = torch.stack(returns)  # [time_steps, chromosomes_size]
        
        # Annualized return
        total_return = (1 + returns_tensor).prod(dim=0) - 1
        annualized_return = (1 + total_return) ** (self.lookback_period / len(returns)) - 1
        
        # Max drawdown (from current max_drawdown tracking)
        max_dd = torch.clamp(self.max_drawdown, min=1e-8)  # Avoid division by zero
        
        calmar_ratios = annualized_return / max_dd
        
        return calmar_ratios
    
    def _calculate_sortino_ratio(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Calculate Sortino ratio (uses downside deviation instead of total volatility).
        
        Args:
            returns: [time_steps, chromosomes_size] tensor of period returns
            
        Returns:
            sortino_ratios: [chromosomes_size] tensor of Sortino ratios
        """
        if len(returns) < 2:
            return torch.zeros(self.chromosomes_size, device=self.device)
        
        returns_tensor = torch.stack(returns)  # [time_steps, chromosomes_size]
        
        # Calculate excess returns
        period_rf_rate = self.risk_free_rate / self.lookback_period
        excess_returns = returns_tensor - period_rf_rate
        
        # Downside deviation (only negative returns)
        negative_returns = torch.where(excess_returns < 0, excess_returns, torch.zeros_like(excess_returns))
        downside_std = torch.sqrt(torch.mean(negative_returns ** 2, dim=0))
        
        # Mean excess return
        mean_excess = excess_returns.mean(dim=0)
        
        # Sortino ratio with numerical stability
        sortino_ratios = torch.where(
            downside_std > 1e-8,
            mean_excess / downside_std * np.sqrt(self.lookback_period),  # Annualized
            torch.zeros_like(mean_excess)
        )
        
        return sortino_ratios
    
    def get_multi_objective_metrics(self, minimum_date: int = 40) -> np.ndarray:
        """
        Calculate multi-objective metrics based on configured objectives.
        
        Args:
            minimum_date (int): Minimum number of trading periods required for valid metrics
            
        Returns:
            objectives: [chromosomes_size, num_objectives] array of objective values
        """
        # Get base metrics from parent class
        base_metrics = super().get_final_metrics(minimum_date)
        
        # base_metrics columns: [mean_returns, profit_factors, win_rates, max_drawdown, compound_value, trade_win_rates]
        mean_returns = torch.tensor(base_metrics[:, 0], device=self.device)
        profit_factors = torch.tensor(base_metrics[:, 1], device=self.device)
        win_rates = torch.tensor(base_metrics[:, 2], device=self.device)
        max_drawdown = torch.tensor(base_metrics[:, 3], device=self.device)
        compound_value = torch.tensor(base_metrics[:, 4], device=self.device)
        trade_win_rates = torch.tensor(base_metrics[:, 5], device=self.device)
        
        # Calculate additional advanced metrics if we have return history
        if len(self.returns_history) >= 2:
            sharpe_ratios = self._calculate_sharpe_ratio(self.returns_history)
            volatilities = self._calculate_volatility(self.returns_history)
            calmar_ratios = self._calculate_calmar_ratio(self.returns_history)
            sortino_ratios = self._calculate_sortino_ratio(self.returns_history)
        else:
            # Fallback for insufficient data
            sharpe_ratios = torch.zeros_like(mean_returns)
            volatilities = torch.zeros_like(mean_returns)
            calmar_ratios = torch.zeros_like(mean_returns)
            sortino_ratios = torch.zeros_like(mean_returns)
        
        # Build objective dictionary
        available_objectives = {
            'mean_return': mean_returns,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratios,
            'win_rate': win_rates,
            'profit_factor': profit_factors,
            'compound_value': compound_value,
            'trade_win_rate': trade_win_rates,
            'volatility': volatilities,
            'calmar_ratio': calmar_ratios,
            'sortino_ratio': sortino_ratios
        }
        
        # Select and arrange objectives according to configuration
        objectives_list = []
        for i, obj_name in enumerate(self.objective_names):
            if obj_name not in available_objectives:
                raise ValueError(f"Unknown objective: {obj_name}. Available: {list(available_objectives.keys())}")
            
            objective_values = available_objectives[obj_name]
            
            # Apply minimization/maximization transformation
            if self.minimize_objectives[i]:
                # For minimization objectives, use as-is (assuming lower is better)
                objectives_list.append(objective_values)
            else:
                # For maximization objectives, negate for minimization framework
                objectives_list.append(-objective_values)
        
        # Stack objectives
        objectives_tensor = torch.stack(objectives_list, dim=1)  # [chromosomes_size, num_objectives]
        
        # Handle invalid individuals (same logic as base class)
        invalid_mask = torch.tensor(base_metrics[:, 0] == -1e9, device=self.device)
        
        # Set invalid individuals to worst possible values for each objective
        for i in range(len(self.objective_names)):
            if self.minimize_objectives[i]:
                # For minimization: set to large positive value
                objectives_tensor[invalid_mask, i] = 1e9
            else:
                # For maximization (negated): set to large positive value (worst for negated)
                objectives_tensor[invalid_mask, i] = 1e9
        
        return objectives_tensor.cpu().numpy()
    
    def get_objective_names(self) -> List[str]:
        """Get the names of configured objectives."""
        return self.objective_names.copy()
    
    def get_minimization_flags(self) -> List[bool]:
        """Get the minimization flags for each objective."""
        return self.minimize_objectives.copy()


def multi_objective_fitness_fn(
    population,
    data: pd.DataFrame,
    all_feature_names: list,
    adj_offsets: torch.Tensor,
    adj_indices: torch.Tensor,
    start_data_cnt: int,
    stop_data_cnt: int,
    device: str,
    evaluation_config: dict
) -> np.ndarray:
    """
    Multi-objective fitness function wrapper.
    
    Args:
        population: GATreePop population
        data: Market data DataFrame
        all_feature_names: List of feature names
        adj_offsets: Adjacency list offsets
        adj_indices: Adjacency list indices
        start_data_cnt: Start index for evaluation
        stop_data_cnt: End index for evaluation
        device: Device for computations
        evaluation_config: Evaluation configuration
        
    Returns:
        objectives: [population_size, num_objectives] array of objective values
    """
    from .predictor import predict_population_cuda
    from tqdm import tqdm
    
    env_config = evaluation_config['simulation_env']
    objectives_config = evaluation_config.get('multi_objectives', {
        'objectives': ['mean_return', 'max_drawdown', 'sharpe_ratio', 'win_rate'],
        'minimize': [False, True, False, False],
        'risk_free_rate': 0.0,
        'volatility_lookbook': 30
    })
    
    # Create multi-objective environment
    environment = MultiObjectiveTradingEnvironment(
        chromosomes_size=population.pop_size,
        env_config=env_config,
        objectives_config=objectives_config,
        device=device
    )
    
    # Position mapping for CUDA prediction
    INT_TO_POSITION_MAP = {0: 'HOLD', 1: 'SHORT', 2: 'LONG'}
    
    # Time-driven simulation loop
    pbar = tqdm(range(start_data_cnt, stop_data_cnt - 1), desc="Multi-Objective Fitness Simulation")
    feature_data = data[all_feature_names]
    
    for entry_index in pbar:
        market_data_row = data.iloc[entry_index]
        next_open_price = torch.tensor(data['Open'].iloc[entry_index + 1], dtype=torch.float32, device=device)
        feature_values_series = feature_data.iloc[entry_index]
        
        # Convert positions for CUDA prediction
        current_positions_str = [INT_TO_POSITION_MAP[pos.item()] for pos in environment.pos_list]
        
        # Predict actions
        predicted_actions = predict_population_cuda(
            population=population,
            feature_values=feature_values_series,
            current_positions=current_positions_str,
            adj_offsets=adj_offsets,
            adj_indices=adj_indices,
            device=device
        )
        
        if predicted_actions is None:
            raise RuntimeError("CUDA prediction failed. Check if gatree_cuda module is compiled and available.")
        
        # Execute step
        environment.step(market_data_row, next_open_price, predicted_actions)
    
    pbar.close()
    
    # Get multi-objective metrics
    minimum_trades = evaluation_config.get('fitness_function', {}).get('minimum_trades', 40)
    objectives = environment.get_multi_objective_metrics(minimum_date=minimum_trades)
    
    return objectives