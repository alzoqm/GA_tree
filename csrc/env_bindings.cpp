// csrc/env_bindings.cpp
// Python bindings for trading environment CUDA kernels

#include <torch/extension.h>
#include "trading_env_kernel.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA kernels for trading environment actions execution";

    m.def(
        "prepare_and_deduplicate_actions",
        &prepare_and_deduplicate_actions_cuda,
        "Prepare actions by applying duplicate detection on GPU",
        py::arg("actions"),
        py::arg("raw_actions"),
        py::arg("prev_actions_raw"),
        py::arg("prev_applied_mask"),
        py::arg("dedup_exclude_types"),
        py::arg("compare_cols"),
        py::arg("rtol"),
        py::arg("atol"),
        py::arg("require_applied"),
        py::arg("action_not_found"),
        py::arg("last_action_node_idx") = py::none(),
        py::arg("node_attempt_count") = py::none(),
        py::arg("node_stall_count") = py::none(),
        py::arg("max_nodes") = 0
    );

    m.def(
        "update_dedup_buffers_after_exec",
        &update_dedup_buffers_after_exec_cuda,
        "Update deduplication buffers after executing actions",
        py::arg("actions"),
        py::arg("raw_actions"),
        py::arg("pos_before"),
        py::arg("er_before"),
        py::arg("pos_after"),
        py::arg("er_after"),
        py::arg("prev_actions_raw"),
        py::arg("prev_applied_mask"),
        py::arg("cols_to_copy"),
        py::arg("eps"),
        py::arg("action_new_long"),
        py::arg("action_new_short"),
        py::arg("action_close_all"),
        py::arg("action_close_partial"),
        py::arg("action_add_position"),
        py::arg("action_flip_position")
    );

    m.def(
        "execute_actions_batch",
        &execute_actions_batch_cuda,
        "Executes trading actions for a batch (thread-per-individual)",
        py::arg("actions"),
        py::arg("pos_list"),
        py::arg("price_list"),
        py::arg("leverage_ratio"),
        py::arg("enter_ratio"),
        py::arg("additional_count"),
        py::arg("profit"),
        py::arg("trade_open"),
        py::arg("trade_realized_pnl"),
        py::arg("trade_total_count"),
        py::arg("trade_win_count"),
        py::arg("trade_holding_sum"),
        py::arg("trade_holding_avg"),
        py::arg("curr_win_streak"),
        py::arg("curr_loss_streak"),
        py::arg("max_win_streak"),
        py::arg("max_loss_streak"),
        py::arg("holding_period"),
        py::arg("buy_exec_price"),
        py::arg("sell_exec_price"),
        py::arg("taker_fee_rate"),
        py::arg("min_entry_ratio"),
        py::arg("min_leverage"),
        py::arg("max_leverage"),
        py::arg("min_additional_enter_ratio"),
        py::arg("max_additional_entries")
    );

    m.def(
        "execute_actions_with_attrib",
        &execute_actions_with_attrib_batch_cuda,
        "Executes actions and updates lot ledger and node-level attribution",
        py::arg("actions"),
        py::arg("action_node_idx"),
        py::arg("pos_list"),
        py::arg("price_list"),
        py::arg("leverage_ratio"),
        py::arg("enter_ratio"),
        py::arg("additional_count"),
        py::arg("profit"),
        py::arg("trade_open"),
        py::arg("trade_realized_pnl"),
        py::arg("trade_total_count"),
        py::arg("trade_win_count"),
        py::arg("trade_holding_sum"),
        py::arg("trade_holding_avg"),
        py::arg("curr_win_streak"),
        py::arg("curr_loss_streak"),
        py::arg("max_win_streak"),
        py::arg("max_loss_streak"),
        py::arg("holding_period"),
        py::arg("lot_open"),
        py::arg("lot_node_idx"),
        py::arg("lot_side"),
        py::arg("lot_entry"),
        py::arg("lot_exposure"),
        py::arg("lot_open_step"),
        py::arg("node_attempt_count"),
        py::arg("node_apply_count"),
        py::arg("node_noop_count"),
        py::arg("node_win_count"),
        py::arg("node_realized_pnl_sum"),
        py::arg("node_exposure_sum"),
        py::arg("current_step"),
        py::arg("buy_exec_price"),
        py::arg("sell_exec_price"),
        py::arg("taker_fee_rate"),
        py::arg("min_entry_ratio"),
        py::arg("min_leverage"),
        py::arg("max_leverage"),
        py::arg("min_additional_enter_ratio"),
        py::arg("max_additional_entries")
    );

    m.def(
        "update_holding_periods",
        &update_holding_periods_cuda,
        "Increment or reset holding periods depending on open positions",
        py::arg("pos_list"),
        py::arg("holding_period")
    );

    m.def(
        "accumulate_realized_stats",
        &accumulate_realized_stats_cuda,
        "Accumulate realized PnL and stats for a batch",
        py::arg("profit"),
        py::arg("realized_cum"),
        py::arg("sum_returns"),
        py::arg("count_returns"),
        py::arg("sum_sq_returns"),
        py::arg("sum_neg"),
        py::arg("count_neg"),
        py::arg("sum_sq_neg"),
        py::arg("total_profit_agg"),
        py::arg("total_loss_agg"),
        py::arg("count_wins"),
        py::arg("compound_value")
    );

    m.def(
        "check_liquidation_batch",
        &check_liquidation_batch_cuda,
        "Check and apply liquidations for a batch (thread-per-individual)",
        py::arg("pos_list"),
        py::arg("price_list"),
        py::arg("leverage_ratio"),
        py::arg("enter_ratio"),
        py::arg("additional_count"),
        py::arg("holding_period"),
        py::arg("profit"),
        py::arg("trade_open"),
        py::arg("trade_realized_pnl"),
        py::arg("trade_total_count"),
        py::arg("trade_win_count"),
        py::arg("trade_holding_sum"),
        py::arg("trade_holding_avg"),
        py::arg("curr_win_streak"),
        py::arg("curr_loss_streak"),
        py::arg("max_win_streak"),
        py::arg("max_loss_streak"),
        py::arg("mark_low"),
        py::arg("mark_high"),
        py::arg("maintenance_margin_rate"),
        py::arg("liq_clearance_fee_rate")
    );

    m.def(
        "check_liquidation_with_attrib",
        &check_liquidation_with_attrib_cuda,
        "Check and apply liquidations with node-level attribution (thread-per-individual)",
        py::arg("pos_list"),
        py::arg("price_list"),
        py::arg("leverage_ratio"),
        py::arg("enter_ratio"),
        py::arg("additional_count"),
        py::arg("holding_period"),
        py::arg("profit"),
        py::arg("trade_open"),
        py::arg("trade_realized_pnl"),
        py::arg("trade_total_count"),
        py::arg("trade_win_count"),
        py::arg("trade_holding_sum"),
        py::arg("trade_holding_avg"),
        py::arg("curr_win_streak"),
        py::arg("curr_loss_streak"),
        py::arg("max_win_streak"),
        py::arg("max_loss_streak"),
        py::arg("lot_open"),
        py::arg("lot_node_idx"),
        py::arg("lot_side"),
        py::arg("lot_entry"),
        py::arg("lot_exposure"),
        py::arg("lot_open_step"),
        py::arg("node_win_count"),
        py::arg("node_realized_pnl_sum"),
        py::arg("node_exposure_sum"),
        py::arg("mark_low"),
        py::arg("mark_high"),
        py::arg("maintenance_margin_rate"),
        py::arg("liq_clearance_fee_rate")
    );

    m.def(
        "compute_unrealized_pnl",
        &compute_unrealized_pnl_cuda,
        "Compute unrealized PnL for a batch",
        py::arg("pos_list"),
        py::arg("price_list"),
        py::arg("leverage_ratio"),
        py::arg("enter_ratio"),
        py::arg("mark_t")
    );

    m.def(
        "compute_equity_batch",
        [](torch::Tensor pos_list,
           torch::Tensor price_list,
           torch::Tensor leverage_ratio,
           torch::Tensor enter_ratio,
           torch::Tensor realized_cum,
           float mark_t,
           float equity_base) {
            auto equity_out = torch::empty_like(realized_cum);
            compute_equity_batch_cuda(
                pos_list, price_list, leverage_ratio, enter_ratio, realized_cum,
                mark_t, equity_base, equity_out
            );
            return equity_out;
        },
        "Compute equity (realized + unrealized) for a batch",
        py::arg("pos_list"),
        py::arg("price_list"),
        py::arg("leverage_ratio"),
        py::arg("enter_ratio"),
        py::arg("realized_cum"),
        py::arg("mark_t"),
        py::arg("equity_base")
    );
}
