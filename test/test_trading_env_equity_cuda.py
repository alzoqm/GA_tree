# /test/test_trading_env_equity_cuda.py
"""Validation and performance tests for CUDA equity/drawdown updates."""

import math
import time

import pytest
import torch

from training.trading_env import TradingEnvironment


def _require_cuda_kernel():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required for this test")
    try:
        import gatree_cuda_compat as gatree_cuda  # noqa: WPS433 (import inside helper)
    except ImportError:
        pytest.skip("gatree_cuda_compat extension is not available")
    if getattr(gatree_cuda, "update_equity_and_drawdown", None) is None:
        pytest.skip("update_equity_and_drawdown CUDA kernel is not exposed")


def _make_env_config():
    return {
        "taker_fee_rate": 0.0005,
        "maintenance_margin_rate": 0.005,
        "equity_base": 1000.0,
        "slippage": {"mode": "fixed", "fixed_rate": 0.0},
        "enable_action_dedup": False,
    }


def _reset_state(env: TradingEnvironment, realized, total_unreal, running_max, max_drawdown, profit):
    if not str(env.device).startswith("cuda"):
        raise RuntimeError("TradingEnvironment must be initialized on CUDA for these tests")
    device = env.device
    env.realized_cum = realized.to(device).clone()
    env.cum_sum = torch.full_like(env.realized_cum, float(env.equity_base))
    env.running_max = running_max.to(device).clone()
    env.max_drawdown = max_drawdown.to(device).clone()
    env.profit = profit.to(device).clone()
    env.last_timestamp = -1
    return total_unreal.to(device).clone()


def _reference_update(realized, total_unreal, running_max, max_drawdown, equity_base):
    equity = equity_base + realized + total_unreal
    updated_running_max = torch.maximum(running_max, equity)
    drawdown = updated_running_max - equity
    denom = torch.clamp(updated_running_max, min=1e-8)
    ratio = torch.where(
        updated_running_max > 0,
        drawdown / denom,
        torch.zeros_like(drawdown),
    )
    ratio = ratio.clamp(0.0, 1.0)
    updated_max_drawdown = torch.maximum(max_drawdown, ratio)
    return equity, updated_running_max, updated_max_drawdown


@pytest.mark.cuda
def test_update_equity_and_drawdown_cuda_matches_cpu():
    _require_cuda_kernel()

    torch.manual_seed(42)
    size = 2048
    env_config = _make_env_config()
    env = TradingEnvironment(size, env_config, device="cuda")

    realized = torch.randn(size, dtype=torch.float32)
    total_unreal = torch.randn(size, dtype=torch.float32) * 0.5
    running_max = torch.full((size,), env_config["equity_base"], dtype=torch.float32)
    running_max += torch.rand(size, dtype=torch.float32) * 250.0
    running_max[::11] = 0.0  # exercise zero-peak branch
    max_drawdown = torch.rand(size, dtype=torch.float32)
    profit = torch.randn(size, dtype=torch.float32)

    total_unreal_gpu = _reset_state(env, realized, total_unreal, running_max, max_drawdown, profit)

    timestamp = 1_697_000_000

    expected_cum_sum, expected_running_max, expected_max_drawdown = _reference_update(
        realized,
        total_unreal,
        running_max,
        max_drawdown,
        env_config["equity_base"],
    )

    env._update_equity_and_drawdown(total_unreal_gpu, timestamp)
    torch.cuda.synchronize()

    assert torch.allclose(env.cum_sum.cpu(), expected_cum_sum, atol=1e-6)
    assert torch.allclose(env.running_max.cpu(), expected_running_max, atol=1e-6)
    assert torch.allclose(env.max_drawdown.cpu(), expected_max_drawdown, atol=1e-6)
    assert torch.all(env.profit.abs() == 0)
    assert env.last_timestamp == timestamp


@pytest.mark.cuda
def test_update_equity_and_drawdown_cuda_speed_advantage():
    _require_cuda_kernel()

    torch.manual_seed(7)
    size = 1_000_000
    env_config = _make_env_config()
    env = TradingEnvironment(size, env_config, device="cuda")

    realized = torch.randn(size, dtype=torch.float32)
    total_unreal = torch.randn(size, dtype=torch.float32)
    running_max = torch.full((size,), env_config["equity_base"], dtype=torch.float32) + torch.rand(size, dtype=torch.float32) * 100.0
    max_drawdown = torch.rand(size, dtype=torch.float32)
    profit = torch.randn(size, dtype=torch.float32)
    timestamp = 42

    # Warm-up CUDA kernel once to avoid initialization overhead
    total_unreal_gpu = _reset_state(env, realized, total_unreal, running_max, max_drawdown, profit)
    env._update_equity_and_drawdown(total_unreal_gpu, timestamp)
    torch.cuda.synchronize()

    # Reset states after warm-up
    _reset_state(env, realized, total_unreal, running_max, max_drawdown, profit)

    cpu_times = []
    for _ in range(3):
        start = time.perf_counter()
        _reference_update(
            realized,
            total_unreal,
            running_max,
            max_drawdown,
            env_config["equity_base"],
        )
        cpu_times.append(time.perf_counter() - start)

    gpu_times = []
    for _ in range(3):
        total_unreal_gpu_copy = _reset_state(
            env, realized, total_unreal, running_max, max_drawdown, profit
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        env._update_equity_and_drawdown(total_unreal_gpu_copy, timestamp)
        torch.cuda.synchronize()
        gpu_times.append(time.perf_counter() - start)

    best_cpu = min(cpu_times)
    best_gpu = min(gpu_times)

    assert not math.isclose(best_cpu, 0.0), "CPU timing should be non-zero"
    assert best_gpu < best_cpu, "CUDA path should outperform CPU baseline"
