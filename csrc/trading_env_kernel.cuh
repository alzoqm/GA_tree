// csrc/trading_env_kernel.cuh
#pragma once

#include <torch/extension.h>

// Host launcher exposed to Python
void execute_actions_batch_cuda(
    torch::Tensor actions,
    torch::Tensor pos_list,
    torch::Tensor price_list,
    torch::Tensor leverage_ratio,
    torch::Tensor enter_ratio,
    torch::Tensor additional_count,
    torch::Tensor profit,
    torch::Tensor trade_open,
    torch::Tensor trade_realized_pnl,
    torch::Tensor trade_total_count,
    torch::Tensor trade_win_count,
    torch::Tensor trade_holding_sum,
    torch::Tensor trade_holding_avg,
    torch::Tensor curr_win_streak,
    torch::Tensor curr_loss_streak,
    torch::Tensor max_win_streak,
    torch::Tensor max_loss_streak,
    torch::Tensor holding_period,
    float buy_exec_price,
    float sell_exec_price,
    float taker_fee_rate,
    float min_entry_ratio,
    int64_t min_leverage,
    int64_t max_leverage,
    float min_additional_enter_ratio,
    int64_t max_additional_entries);

// Host launcher to accumulate realized PnL and related stats per individual
void accumulate_realized_stats_cuda(
    torch::Tensor profit,
    torch::Tensor realized_cum,
    torch::Tensor sum_returns,
    torch::Tensor count_returns,
    torch::Tensor sum_sq_returns,
    torch::Tensor sum_neg,
    torch::Tensor count_neg,
    torch::Tensor sum_sq_neg,
    torch::Tensor total_profit_agg,
    torch::Tensor total_loss_agg,
    torch::Tensor count_wins,
    torch::Tensor compound_value);
