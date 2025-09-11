#!/usr/bin/env python3
"""
Enhanced test script for the improved dynamic slippage implementation with column auto-detection.
Tests various multi-timeframe column naming patterns.
"""

import pandas as pd
import numpy as np
import re
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class MockTradingEnvironment:
    """Mock TradingEnvironment for testing column detection logic."""
    
    def __init__(self, slippage_config):
        self.slippage_mode = slippage_config.get('mode', 'fixed')
        
        if self.slippage_mode == 'fixed':
            self.fixed_slippage_rate = slippage_config.get('fixed_rate', 0.0005)
        elif self.slippage_mode == 'relative_volatility':
            # Initialize fixed rate for fallback
            self.fixed_slippage_rate = 0.0005
            self.base_slippage_rate = slippage_config.get('base_rate', 0.0001)
            self.atr_ma_window = slippage_config.get('atr_ma_window', 50)
            self.min_slippage = slippage_config.get('min_slippage', 0.00005)
            self.max_slippage = slippage_config.get('max_slippage', 0.002)
            
            # ATR column configuration
            self.atr_column_config = slippage_config.get('atr_column')
            self.atr_ma_column_config = slippage_config.get('atr_ma_column')
            self.atr_pattern = slippage_config.get('atr_pattern', r'ATR_\d+.*')
            self.atr_ma_pattern = slippage_config.get('atr_ma_pattern', r'ATR_\d+.*_MA_\d+')
            
            # Initialize detection state
            self.atr_column = None
            self.atr_ma_column = None
            self._columns_detected = False

    def _detect_atr_columns(self, market_data_row):
        """Detect available ATR and ATR-MA columns with enhanced logic."""
        available_columns = list(market_data_row.index)
        
        # 1. Use user-specified columns if available
        if self.atr_column_config and self.atr_ma_column_config:
            if (self.atr_column_config in available_columns and 
                self.atr_ma_column_config in available_columns):
                print(f"✓ Using user-specified ATR columns: {self.atr_column_config}, {self.atr_ma_column_config}")
                return self.atr_column_config, self.atr_ma_column_config
            else:
                print(f"⚠️ User-specified columns not found. Falling back to auto-detection.")
        
        # 2. Pattern matching for auto-detection
        atr_candidates = [col for col in available_columns if re.match(self.atr_pattern, col)]
        atr_ma_candidates = [col for col in available_columns if re.match(self.atr_ma_pattern, col)]
        
        if not atr_candidates:
            raise ValueError(
                f"No ATR columns found matching pattern '{self.atr_pattern}'. "
                f"Available ATR-like columns: {[col for col in available_columns if 'ATR' in col]}"
            )
        
        if not atr_ma_candidates:
            raise ValueError(
                f"No ATR-MA columns found matching pattern '{self.atr_ma_pattern}'. "
                f"Available ATR-like columns: {[col for col in available_columns if 'ATR' in col]}"
            )
        
        # 3. Find optimal ATR/ATR-MA pairs
        best_atr = None
        best_atr_ma = None
        
        # Priority windows for ATR (common periods)
        preferred_windows = [14, 7, 21, 28, 20, 10]
        
        for window in preferred_windows:
            for atr_col in atr_candidates:
                if f'_{window}' in atr_col or atr_col.endswith(str(window)):
                    # Look for matching MA column with exact prefix matching
                    matching_ma = [ma_col for ma_col in atr_ma_candidates 
                                   if ma_col.startswith(atr_col + '_MA_')]
                    
                    # Prefer MA columns with the configured MA window
                    preferred_ma = [ma_col for ma_col in matching_ma 
                                    if f'_MA_{self.atr_ma_window}' in ma_col]
                    
                    if preferred_ma:
                        best_atr = atr_col
                        best_atr_ma = preferred_ma[0]
                        break
                    elif matching_ma:
                        # Use first matching MA if preferred not found
                        best_atr = atr_col
                        best_atr_ma = matching_ma[0]
                        break
            
            if best_atr and best_atr_ma:
                break
        
        # 4. Fallback: try any matching pair
        if not best_atr or not best_atr_ma:
            for atr_col in sorted(atr_candidates):
                # Look for MA column with exact prefix matching
                matching_ma = [ma_col for ma_col in atr_ma_candidates 
                               if ma_col.startswith(atr_col + '_MA_')]
                if matching_ma:
                    best_atr = atr_col
                    best_atr_ma = matching_ma[0]
                    break
        
        if not best_atr or not best_atr_ma:
            raise ValueError(
                f"Could not find matching ATR/ATR-MA column pair. "
                f"ATR candidates: {atr_candidates}, ATR-MA candidates: {atr_ma_candidates}"
            )
        
        print(f"🔍 Auto-detected ATR columns: {best_atr} → {best_atr_ma}")
        return best_atr, best_atr_ma

    def _calculate_dynamic_slippage(self, market_data_row):
        """Calculate dynamic slippage with auto-detection."""
        if self.slippage_mode != 'relative_volatility':
            return self.fixed_slippage_rate
        
        # Detect columns on first call
        if not self._columns_detected:
            try:
                self.atr_column, self.atr_ma_column = self._detect_atr_columns(market_data_row)
                self._columns_detected = True
            except ValueError as e:
                print(f"❌ ATR column detection failed: {e}")
                print("📛 Falling back to fixed slippage mode.")
                return self.fixed_slippage_rate
        
        try:
            # Extract ATR values
            current_atr = market_data_row.get(self.atr_column)
            atr_ma = market_data_row.get(self.atr_ma_column)
            
            # Data validation
            if current_atr is None or atr_ma is None:
                print(f"⚠️ Missing data in columns ({self.atr_column}, {self.atr_ma_column}). Using base rate.")
                return self.base_slippage_rate
            
            if pd.isna(current_atr) or pd.isna(atr_ma) or atr_ma <= 0:
                return self.base_slippage_rate
            
            # Calculate volatility ratio and dynamic slippage
            volatility_ratio = float(current_atr) / float(atr_ma)
            dynamic_slippage = self.base_slippage_rate * volatility_ratio
            final_slippage = max(self.min_slippage, min(self.max_slippage, dynamic_slippage))
            
            return final_slippage
            
        except Exception as e:
            print(f"❌ Error in slippage calculation: {e}. Using base rate.")
            return self.base_slippage_rate


def test_column_detection_scenarios():
    """Test various column naming scenarios."""
    print("🧪 Testing Enhanced Column Detection for Dynamic Slippage")
    print("=" * 60)
    
    # Test configuration
    base_config = {
        'mode': 'relative_volatility',
        'base_rate': 0.0001,
        'atr_ma_window': 50,
        'min_slippage': 0.00005,
        'max_slippage': 0.002,
        'atr_pattern': r'ATR_\d+.*',
        'atr_ma_pattern': r'ATR_\d+.*_MA_\d+'
    }
    
    # Test scenarios with different column naming patterns
    scenarios = [
        {
            'name': "User's Multi-timeframe Pattern",
            'columns': ['ATR_7_5m', 'ATR_7_5m_MA_14', 'ATR_14_5m', 'ATR_14_5m_MA_50', 'Close', 'Volume'],
            'config': {**base_config, 'atr_ma_window': 14}
        },
        {
            'name': "Standard Naming Pattern", 
            'columns': ['ATR_14', 'ATR_14_MA_50', 'ATR_28', 'ATR_28_MA_50', 'Close', 'Volume'],
            'config': base_config
        },
        {
            'name': "Mixed Timeframe Pattern",
            'columns': ['ATR_14_1h', 'ATR_14_1h_MA_50', 'ATR_7_5m', 'ATR_7_5m_MA_20', 'Close', 'Volume'],
            'config': base_config
        },
        {
            'name': "User-Specified Columns",
            'columns': ['ATR_custom', 'ATR_custom_MA_30', 'ATR_14', 'ATR_14_MA_50', 'Close', 'Volume'],
            'config': {**base_config, 'atr_column': 'ATR_custom', 'atr_ma_column': 'ATR_custom_MA_30'}
        },
        {
            'name': "Missing ATR-MA Columns",
            'columns': ['ATR_14', 'ATR_28', 'Close', 'Volume'],
            'config': base_config
        },
        {
            'name': "No ATR Columns",
            'columns': ['Close', 'Volume', 'RSI_14', 'MACD_12_26'],
            'config': base_config
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['name']}")
        print(f"   Available columns: {scenario['columns']}")
        
        # Create mock data
        data = {col: np.random.uniform(0.1, 0.3, 1) for col in scenario['columns']}
        # Add realistic ATR-MA values (should be close to ATR values)
        for col in scenario['columns']:
            if '_MA_' in col:
                base_atr_col = col.split('_MA_')[0]
                if base_atr_col in data:
                    data[col] = data[base_atr_col] * np.random.uniform(0.8, 1.2)  # MA usually smooths
        
        market_data = pd.Series(data)
        
        try:
            env = MockTradingEnvironment(scenario['config'])
            slippage = env._calculate_dynamic_slippage(market_data)
            
            if env._columns_detected:
                print(f"   ✅ SUCCESS: Detected columns {env.atr_column} → {env.atr_ma_column}")
                print(f"   💰 Dynamic slippage: {slippage:.6f} ({slippage*100:.4f}%)")
            else:
                print(f"   📛 Used fallback fixed slippage: {slippage:.6f} ({slippage*100:.4f}%)")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
        
        print("   " + "-" * 50)
    
    print("\n✅ Column detection tests completed!")


def test_pattern_matching():
    """Test regex pattern matching for different column formats."""
    print("\n🔍 Testing Regex Pattern Matching")
    print("=" * 40)
    
    patterns = {
        'atr_pattern': r'ATR_\d+.*',
        'atr_ma_pattern': r'ATR_\d+.*_MA_\d+'
    }
    
    test_columns = [
        'ATR_14',
        'ATR_7_5m', 
        'ATR_14_1h',
        'ATR_28_4h',
        'ATR_14_MA_50',
        'ATR_7_5m_MA_14',
        'ATR_14_1h_MA_50',
        'ATR_custom',
        'NATR_14',
        'Close',
        'Volume'
    ]
    
    print("Testing ATR pattern matching:")
    for col in test_columns:
        atr_match = re.match(patterns['atr_pattern'], col)
        atr_ma_match = re.match(patterns['atr_ma_pattern'], col)
        
        status = "✓" if atr_match else "✗"
        ma_status = "✓" if atr_ma_match else "✗"
        
        print(f"  {col:<20} | ATR: {status} | ATR-MA: {ma_status}")
    
    print("\n✅ Pattern matching tests completed!")


if __name__ == "__main__":
    test_column_detection_scenarios()
    test_pattern_matching()
    
    print("\n🚀 Enhanced Dynamic Slippage Implementation Ready!")
    print("\n📝 Summary of improvements:")
    print("  • ✅ Automatic column detection with multiple naming patterns")
    print("  • ✅ User-configurable column names with fallback")
    print("  • ✅ Priority-based column selection (prefers ATR_14, ATR_7, etc.)")
    print("  • ✅ Enhanced error messages with available column listings")
    print("  • ✅ Graceful fallback to fixed slippage when detection fails")
    print("  • ✅ Support for multi-timeframe naming (ATR_7_5m, ATR_14_1h, etc.)")