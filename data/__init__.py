# /data/__init__.py
"""Data subpackage.

Avoid importing heavy optional dependencies (e.g., TA-Lib) at package import
time. Import submodules explicitly where needed, e.g.:

    from data import ta_lib_feature_generator as talib_feat

This keeps lightweight utilities (like classification helpers) usable even
when TA-Lib is not installed.
"""
