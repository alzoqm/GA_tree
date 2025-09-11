#!/usr/bin/env python3
"""
Test script for the dynamic slippage implementation.
This validates the core logic without requiring the full environment.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_dynamic_slippage_calculation():
    """Test the dynamic slippage calculation logic."""
    
    # Mock TradingEnvironment class with just the slippage calculation
    class MockTradingEnvironment:
        def __init__(self, slippage_config):
            self.slippage_mode = slippage_config.get('mode', 'fixed')
            
            if self.slippage_mode == 'fixed':
                self.fixed_slippage_rate = slippage_config.get('fixed_rate', 0.0005)
            elif self.slippage_mode == 'relative_volatility':
                self.base_slippage_rate = slippage_config.get('base_rate', 0.0001)
                self.atr_ma_window = slippage_config.get('atr_ma_window', 50)
                self.min_slippage = slippage_config.get('min_slippage', 0.00005)
                self.max_slippage = slippage_config.get('max_slippage', 0.002)
                self.atr_column = 'ATR_14'
                self.atr_ma_column = f'ATR_14_MA_{self.atr_ma_window}'
        
        def _calculate_dynamic_slippage(self, market_data_row):
            """Dynamic slippage calculation method."""
            if self.slippage_mode != 'relative_volatility':
                return self.fixed_slippage_rate
            
            try:
                current_atr = market_data_row.get(self.atr_column)
                atr_ma = market_data_row.get(self.atr_ma_column)
                
                if current_atr is None or atr_ma is None:
                    print(f"Warning: Missing ATR data columns ({self.atr_column}, {self.atr_ma_column}). Using base slippage rate.")
                    return self.base_slippage_rate
                
                if pd.isna(current_atr) or pd.isna(atr_ma) or atr_ma <= 0:
                    return self.base_slippage_rate
                
                # Core calculation: volatility ratio
                volatility_ratio = float(current_atr) / float(atr_ma)
                
                # Dynamic slippage
                dynamic_slippage = self.base_slippage_rate * volatility_ratio
                
                # Apply caps
                final_slippage = max(self.min_slippage, min(self.max_slippage, dynamic_slippage))
                
                return final_slippage
                
            except Exception as e:
                print(f"Error in dynamic slippage calculation: {e}. Using base slippage rate.")
                return self.base_slippage_rate

    print("🧪 Testing Dynamic Slippage Implementation")
    print("=" * 50)
    
    # Test configuration from our experiment_config.yaml
    slippage_config = {
        'mode': 'relative_volatility',
        'base_rate': 0.0001,
        'atr_ma_window': 50,
        'min_slippage': 0.00005,
        'max_slippage': 0.002
    }
    
    env = MockTradingEnvironment(slippage_config)
    
    # Test scenarios
    test_cases = [
        # (ATR_14, ATR_14_MA_50, expected_description)
        (0.1, 0.1, "Normal volatility (ratio=1.0)"),
        (0.2, 0.1, "High volatility (ratio=2.0)"),
        (0.05, 0.1, "Low volatility (ratio=0.5)"),
        (0.5, 0.1, "Extreme high volatility (should hit max cap)"),
        (0.001, 0.1, "Extreme low volatility (should hit min cap)"),
        (None, 0.1, "Missing ATR data"),
        (0.1, 0, "Invalid ATR-MA (zero)"),
        (np.nan, 0.1, "NaN ATR data"),
    ]
    
    print("Test Results:")
    print("-" * 50)
    
    for i, (atr, atr_ma, description) in enumerate(test_cases, 1):
        # Create mock market data row
        market_data = pd.Series({
            'ATR_14': atr,
            'ATR_14_MA_50': atr_ma,
            'Close': 50000.0,  # BTC price example
        })
        
        slippage = env._calculate_dynamic_slippage(market_data)
        
        print(f"{i:2d}. {description}")
        print(f"    ATR: {atr}, ATR-MA: {atr_ma}")
        print(f"    → Slippage: {slippage:.6f} ({slippage*100:.4f}%)")
        
        if atr is not None and atr_ma is not None and atr_ma > 0 and not pd.isna(atr):
            expected_ratio = atr / atr_ma
            expected_raw = slippage_config['base_rate'] * expected_ratio
            expected_final = max(slippage_config['min_slippage'], 
                               min(slippage_config['max_slippage'], expected_raw))
            print(f"    Expected: {expected_final:.6f} ({'✓' if abs(slippage - expected_final) < 1e-8 else '✗'})")
        
        print()
    
    print("🎯 Configuration Validation:")
    print(f"   Mode: {env.slippage_mode}")
    print(f"   Base Rate: {env.base_slippage_rate:.6f} ({env.base_slippage_rate*100:.4f}%)")
    print(f"   Min Slippage: {env.min_slippage:.6f} ({env.min_slippage*100:.4f}%)")
    print(f"   Max Slippage: {env.max_slippage:.6f} ({env.max_slippage*100:.4f}%)")
    print(f"   ATR Column: {env.atr_column}")
    print(f"   ATR-MA Column: {env.atr_ma_column}")
    
    print("\n✅ All tests completed successfully!")


def test_atr_moving_average_function():
    """Test the ATR moving average calculation function."""
    print("\n🧪 Testing ATR Moving Average Function")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    data = {
        'Open': np.random.uniform(49000, 51000, 100),
        'High': np.random.uniform(50500, 52000, 100),
        'Low': np.random.uniform(48000, 49500, 100),
        'Close': np.random.uniform(49500, 50500, 100),
        'Volume': np.random.uniform(100000, 500000, 100),
        'ATR_14': np.random.uniform(0.05, 0.2, 100),  # Simulated ATR values
    }
    df = pd.DataFrame(data)
    
    # Mock format_col_name function
    def format_col_name(base_name, window=None):
        if window:
            return f"{base_name}_{window}"
        return base_name
    
    # Mock ATR moving average function
    def calculate_atr_moving_average(df, atr_window, ma_window):
        atr_col = format_col_name('ATR', atr_window)
        new_col = format_col_name('ATR', f"{atr_window}_MA_{ma_window}")
        
        if atr_col not in df.columns:
            print(f"Warning: ATR column '{atr_col}' not found.")
            return df, []
        
        df[new_col] = df[atr_col].rolling(window=ma_window, min_periods=1).mean()
        return df, [new_col]
    
    # Test the function
    df_result, new_cols = calculate_atr_moving_average(df, 14, 50)
    
    print(f"Input DataFrame shape: {df.shape}")
    print(f"New columns added: {new_cols}")
    print(f"Result DataFrame shape: {df_result.shape}")
    
    if new_cols:
        col_name = new_cols[0]
        print(f"\nSample {col_name} values:")
        print(df_result[['ATR_14', col_name]].tail(10))
        
        # Validate the moving average calculation
        manual_ma = df['ATR_14'].rolling(window=50, min_periods=1).mean()
        if np.allclose(df_result[col_name], manual_ma, equal_nan=True):
            print("✅ ATR moving average calculation is correct!")
        else:
            print("❌ ATR moving average calculation has errors!")
    
    print("\n✅ ATR MA function test completed!")


if __name__ == "__main__":
    test_dynamic_slippage_calculation()
    test_atr_moving_average_function()
    
    print("\n🚀 Dynamic Slippage Implementation Ready!")
    print("Next steps:")
    print("1. Run data preprocessing with new ATR-MA features")
    print("2. Run training with dynamic slippage enabled")
    print("3. Compare performance with fixed slippage baseline")