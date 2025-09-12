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

// Extended executor with action-node attribution and lot ledger updates
void execute_actions_with_attrib_batch_cuda(
    torch::Tensor actions,            // (B,4)
    torch::Tensor action_node_idx,    // (B,) int32
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
    // Lot ledger (L = N)
    torch::Tensor lot_open,
    torch::Tensor lot_node_idx,
    torch::Tensor lot_side,
    torch::Tensor lot_entry,
    torch::Tensor lot_exposure,
    torch::Tensor lot_open_step,
    // Node accumulators (B,N)
    torch::Tensor node_attempt_count,
    torch::Tensor node_apply_count,
    torch::Tensor node_noop_count,
    torch::Tensor node_win_count,
    torch::Tensor node_realized_pnl_sum,
    torch::Tensor node_exposure_sum,
    // Scalars
    int current_step,
    float buy_exec_price,
    float sell_exec_price,
    float taker_fee_rate,
    float min_entry_ratio,
    int64_t min_leverage,
    int64_t max_leverage,
    float min_additional_enter_ratio,
    int64_t max_additional_entries
);

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

// Host launcher to check liquidation (thread-per-individual)
void check_liquidation_batch_cuda(
    torch::Tensor pos_list,
    torch::Tensor price_list,
    torch::Tensor leverage_ratio,
    torch::Tensor enter_ratio,
    torch::Tensor additional_count,
    torch::Tensor holding_period,
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
    float mark_low,
    float mark_high,
    float maintenance_margin_rate,
    float liq_clearance_fee_rate);

// Host launcher to check liquidation with node-level attribution (thread-per-individual)
void check_liquidation_with_attrib_cuda(
    // State tensors
    torch::Tensor pos_list,
    torch::Tensor price_list,
    torch::Tensor leverage_ratio,
    torch::Tensor enter_ratio,
    torch::Tensor additional_count,
    torch::Tensor holding_period,
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
    // Lot ledger
    torch::Tensor lot_open,
    torch::Tensor lot_node_idx,
    torch::Tensor lot_side,
    torch::Tensor lot_entry,
    torch::Tensor lot_exposure,
    torch::Tensor lot_open_step,
    // Node accumulators (subset used)
    torch::Tensor node_win_count,
    torch::Tensor node_realized_pnl_sum,
    torch::Tensor node_exposure_sum,
    // Scalars
    float mark_low,
    float mark_high,
    float maintenance_margin_rate,
    float liq_clearance_fee_rate);

// Host launcher to compute equity (realized + unrealized) per individual
void compute_equity_batch_cuda(
    torch::Tensor pos_list,
    torch::Tensor price_list,
    torch::Tensor leverage_ratio,
    torch::Tensor enter_ratio,
    torch::Tensor realized_cum,
    float mark_t,
    float equity_base,
    torch::Tensor equity_out);
