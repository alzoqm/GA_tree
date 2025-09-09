// csrc/env_bindings.cpp
// Python bindings for trading environment CUDA kernels

#include <torch/extension.h>
#include "trading_env_kernel.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA kernels for trading environment actions execution";

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
}
