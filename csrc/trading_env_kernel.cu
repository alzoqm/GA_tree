// csrc/trading_env_kernel.cu
// CUDA kernel to execute trading actions per individual (thread-per-individual)

#include "trading_env_kernel.cuh"
#include "constants.h"
#include <cuda_runtime.h>

// Helper: clamp float
__device__ inline float fclamp(float v, float lo, float hi) {
    return fmaxf(lo, fminf(v, hi));
}

// Helper: clamp int64 within bounds
__device__ inline long long iclamp(long long v, long long lo, long long hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Update streaks and holding aggregates on close
__device__ inline void on_trade_close(
    int i,
    float realized_total,
    int32_t* curr_win_streak,
    int32_t* curr_loss_streak,
    int32_t* max_win_streak,
    int32_t* max_loss_streak,
    float* trade_holding_sum,
    const int64_t* holding_period,
    int32_t* trade_total_count,
    float* trade_holding_avg)
{
    const bool is_win = realized_total > 0.0f;
    if (is_win) {
        curr_win_streak[i] += 1;
        if (curr_win_streak[i] > max_win_streak[i]) max_win_streak[i] = curr_win_streak[i];
        curr_loss_streak[i] = 0;
    } else {
        curr_loss_streak[i] += 1;
        if (curr_loss_streak[i] > max_loss_streak[i]) max_loss_streak[i] = curr_loss_streak[i];
        curr_win_streak[i] = 0;
    }
    trade_holding_sum[i] += static_cast<float>(holding_period[i]);
    // avg = sum / count (count already incremented by caller before this)
    const float denom = fmaxf(1.0f, static_cast<float>(trade_total_count[i]));
    trade_holding_avg[i] = trade_holding_sum[i] / denom;
}

__global__ void execute_actions_kernel(
    const float* __restrict__ actions, // (B,4)
    int64_t* __restrict__ pos_list,
    float* __restrict__ price_list,
    int64_t* __restrict__ leverage_ratio,
    float* __restrict__ enter_ratio,
    int64_t* __restrict__ additional_count,
    float* __restrict__ profit,
    bool* __restrict__ trade_open,
    float* __restrict__ trade_realized_pnl,
    int32_t* __restrict__ trade_total_count,
    int32_t* __restrict__ trade_win_count,
    float* __restrict__ trade_holding_sum,
    float* __restrict__ trade_holding_avg,
    int32_t* __restrict__ curr_win_streak,
    int32_t* __restrict__ curr_loss_streak,
    int32_t* __restrict__ max_win_streak,
    int32_t* __restrict__ max_loss_streak,
    const int64_t* __restrict__ holding_period,
    int B,
    float buy_exec_price,
    float sell_exec_price,
    float taker_fee_rate,
    float min_entry_ratio,
    long long min_leverage,
    long long max_leverage,
    float min_additional_enter_ratio,
    long long max_additional_entries)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    const float a0 = actions[i * 4 + 0];
    const float a1 = actions[i * 4 + 1];
    const float a2 = actions[i * 4 + 2];
    // a3 reserved

    const int atype = static_cast<int>(llroundf(a0));
    if (atype == ACTION_NOT_FOUND) return;

    const int64_t pos = pos_list[i];

    // Helpers to read current state fast
    auto fee_apply = [&](float margin, float levf) {
        profit[i] -= margin * levf * taker_fee_rate;
    };

    // Open new positions
    if ((atype == ACTION_NEW_LONG || atype == ACTION_NEW_SHORT) && pos == 0) {
        float ratio = fclamp(a1, min_entry_ratio, 1.0f);
        long long lev = iclamp(static_cast<long long>(llroundf(a2)), min_leverage, max_leverage);
        fee_apply(ratio, static_cast<float>(lev));
        pos_list[i] = (atype == ACTION_NEW_LONG) ? 2 : 1;
        price_list[i] = (atype == ACTION_NEW_LONG) ? buy_exec_price : sell_exec_price;
        leverage_ratio[i] = lev;
        enter_ratio[i] = ratio;
        additional_count[i] = 0;
        trade_open[i] = true;
        trade_realized_pnl[i] = 0.0f;
        return;
    }

    // Close all
    if (atype == ACTION_CLOSE_ALL && pos != 0) {
        const float entry = price_list[i];
        const float levf = static_cast<float>(leverage_ratio[i]);
        const float er = enter_ratio[i];
        float pnl = 0.0f;
        if (pos == 2) {
            pnl = ((sell_exec_price - entry) / (entry + 1e-9f)) * levf * er;
        } else {
            pnl = ((entry - buy_exec_price) / (entry + 1e-9f)) * levf * er;
        }
        const float fee = er * levf * taker_fee_rate;
        const float realized = pnl - fee;
        profit[i] += realized;
        trade_realized_pnl[i] += realized;
        trade_total_count[i] += 1;
        if (trade_realized_pnl[i] > 0.0f) trade_win_count[i] += 1;
        on_trade_close(i, trade_realized_pnl[i], curr_win_streak, curr_loss_streak,
                       max_win_streak, max_loss_streak, trade_holding_sum,
                       holding_period, trade_total_count, trade_holding_avg);
        trade_open[i] = false;
        trade_realized_pnl[i] = 0.0f;
        pos_list[i] = 0;
        price_list[i] = -1.0f;
        leverage_ratio[i] = -1;
        enter_ratio[i] = -1.0f;
        additional_count[i] = 0;
        return;
    }

    // Close partial
    if (atype == ACTION_CLOSE_PARTIAL && pos != 0) {
        const float ratio = a1;
        const float entry = price_list[i];
        const float levf = static_cast<float>(leverage_ratio[i]);
        const float er_cur = enter_ratio[i];
        const float remaining_after = er_cur * fmaxf(0.0f, 1.0f - ratio);
        const bool close_all = (remaining_after <= min_entry_ratio);
        if (close_all) {
            float pnl = 0.0f;
            if (pos == 2) {
                pnl = ((sell_exec_price - entry) / (entry + 1e-9f)) * levf * er_cur;
            } else {
                pnl = ((entry - buy_exec_price) / (entry + 1e-9f)) * levf * er_cur;
            }
            const float fee = er_cur * levf * taker_fee_rate;
            const float realized = pnl - fee;
            profit[i] += realized;
            trade_realized_pnl[i] += realized;
            trade_total_count[i] += 1;
            if (trade_realized_pnl[i] > 0.0f) trade_win_count[i] += 1;
            on_trade_close(i, trade_realized_pnl[i], curr_win_streak, curr_loss_streak,
                           max_win_streak, max_loss_streak, trade_holding_sum,
                           holding_period, trade_total_count, trade_holding_avg);
            trade_open[i] = false;
            trade_realized_pnl[i] = 0.0f;
            pos_list[i] = 0;
            price_list[i] = -1.0f;
            leverage_ratio[i] = -1;
            enter_ratio[i] = -1.0f;
            additional_count[i] = 0;
            return;
        } else {
            // partial close
            const float closed_margin = er_cur * ratio;
            float pnl = 0.0f;
            if (pos == 2) {
                pnl = ((sell_exec_price - entry) / (entry + 1e-9f)) * levf * closed_margin;
            } else {
                pnl = ((entry - buy_exec_price) / (entry + 1e-9f)) * levf * closed_margin;
            }
            const float fee = closed_margin * levf * taker_fee_rate;
            const float realized = pnl - fee;
            profit[i] += realized;
            trade_realized_pnl[i] += realized; // accumulate only
            enter_ratio[i] = er_cur - closed_margin;

            // Decrement additional_count, ensuring it does not go below zero
            additional_count[i] = (additional_count[i] > 0) ? additional_count[i] - 1 : 0;
            
            return;
        }
    }

    // Add position
    if (atype == ACTION_ADD_POSITION && pos != 0 && additional_count[i] < max_additional_entries) {
        const float er = enter_ratio[i];
        const float remaining = fmaxf(0.0f, 1.0f - er);
        if (remaining >= min_additional_enter_ratio) {
            const float desired = fmaxf(a1, min_additional_enter_ratio);
            const float add = fminf(desired, remaining);
            if (add > 0.0f) {
                const float levf = static_cast<float>(leverage_ratio[i]);
                fee_apply(add, levf);
                const float old_price = price_list[i];
                const float new_avg = (old_price * er + ((pos == 2) ? buy_exec_price : sell_exec_price) * add)
                                     / (er + add + 1e-9f);
                price_list[i] = new_avg;
                enter_ratio[i] = er + add;
                additional_count[i] += 1;
                return;
            }
        }
    }

    // Flip position
    if (atype == ACTION_FLIP_POSITION && pos != 0) {
        // Close current fully
        const float entry = price_list[i];
        const float levf_old = static_cast<float>(leverage_ratio[i]);
        const float er_old = enter_ratio[i];
        float pnl = 0.0f;
        if (pos == 2) {
            pnl = ((sell_exec_price - entry) / (entry + 1e-9f)) * levf_old * er_old;
        } else {
            pnl = ((entry - buy_exec_price) / (entry + 1e-9f)) * levf_old * er_old;
        }
        const float fee_close = er_old * levf_old * taker_fee_rate;
        const float realized_close = pnl - fee_close;
        profit[i] += realized_close;
        trade_realized_pnl[i] += realized_close;
        trade_total_count[i] += 1;
        if (trade_realized_pnl[i] > 0.0f) trade_win_count[i] += 1;
        on_trade_close(i, trade_realized_pnl[i], curr_win_streak, curr_loss_streak,
                       max_win_streak, max_loss_streak, trade_holding_sum,
                       holding_period, trade_total_count, trade_holding_avg);
        trade_open[i] = false;
        trade_realized_pnl[i] = 0.0f;

        // Open new in opposite direction
        float ratio = fclamp(a1, min_entry_ratio, 1.0f);
        long long lev = iclamp(static_cast<long long>(llroundf(a2)), min_leverage, max_leverage);
        fee_apply(ratio, static_cast<float>(lev));
        const int64_t new_pos = (pos == 2) ? 1 : 2;
        pos_list[i] = new_pos;
        price_list[i] = (new_pos == 2) ? buy_exec_price : sell_exec_price;
        enter_ratio[i] = ratio;
        leverage_ratio[i] = lev;
        additional_count[i] = 0;
        trade_open[i] = true;
        trade_realized_pnl[i] = 0.0f;
        return;
    }
}

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
    int64_t max_additional_entries)
{
    TORCH_CHECK(actions.is_cuda(), "actions must be CUDA tensor");
    TORCH_CHECK(actions.is_contiguous(), "actions must be contiguous");
    TORCH_CHECK(actions.dim() == 2 && actions.size(1) >= 3, "actions must be (B,>=3)");

    auto check_cuda_contig = [](const torch::Tensor& t, const char* name){
        TORCH_CHECK(t.is_cuda(), name, " must be CUDA tensor");
        TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    };
    check_cuda_contig(pos_list, "pos_list");
    check_cuda_contig(price_list, "price_list");
    check_cuda_contig(leverage_ratio, "leverage_ratio");
    check_cuda_contig(enter_ratio, "enter_ratio");
    check_cuda_contig(additional_count, "additional_count");
    check_cuda_contig(profit, "profit");
    check_cuda_contig(trade_open, "trade_open");
    check_cuda_contig(trade_realized_pnl, "trade_realized_pnl");
    check_cuda_contig(trade_total_count, "trade_total_count");
    check_cuda_contig(trade_win_count, "trade_win_count");
    check_cuda_contig(trade_holding_sum, "trade_holding_sum");
    check_cuda_contig(trade_holding_avg, "trade_holding_avg");
    check_cuda_contig(curr_win_streak, "curr_win_streak");
    check_cuda_contig(curr_loss_streak, "curr_loss_streak");
    check_cuda_contig(max_win_streak, "max_win_streak");
    check_cuda_contig(max_loss_streak, "max_loss_streak");
    check_cuda_contig(holding_period, "holding_period");

    const int B = static_cast<int>(actions.size(0));
    if (B == 0) return;

    const int threads = 256;
    const int blocks = (B + threads - 1) / threads;

    execute_actions_kernel<<<blocks, threads>>>(
        actions.data_ptr<float>(),
        pos_list.data_ptr<int64_t>(),
        price_list.data_ptr<float>(),
        leverage_ratio.data_ptr<int64_t>(),
        enter_ratio.data_ptr<float>(),
        additional_count.data_ptr<int64_t>(),
        profit.data_ptr<float>(),
        trade_open.data_ptr<bool>(),
        trade_realized_pnl.data_ptr<float>(),
        trade_total_count.data_ptr<int32_t>(),
        trade_win_count.data_ptr<int32_t>(),
        trade_holding_sum.data_ptr<float>(),
        trade_holding_avg.data_ptr<float>(),
        curr_win_streak.data_ptr<int32_t>(),
        curr_loss_streak.data_ptr<int32_t>(),
        max_win_streak.data_ptr<int32_t>(),
        max_loss_streak.data_ptr<int32_t>(),
        holding_period.data_ptr<int64_t>(),
        B,
        buy_exec_price,
        sell_exec_price,
        taker_fee_rate,
        min_entry_ratio,
        (long long)min_leverage,
        (long long)max_leverage,
        min_additional_enter_ratio,
        (long long)max_additional_entries
    );
}

// ===================== Extended executor with attribution =====================

__device__ inline int find_free_lot(bool* lot_open_row, int L){
    for (int k = 0; k < L; ++k){ if (!lot_open_row[k]) return k; }
    return -1;
}

__global__ void execute_actions_with_attrib_kernel(
    const float* __restrict__ actions, // (B,4)
    const int*   __restrict__ action_node_idx, // (B,)
    // aggregated state
    int64_t* __restrict__ pos_list,
    float*   __restrict__ price_list,
    int64_t* __restrict__ leverage_ratio,
    float*   __restrict__ enter_ratio,
    int64_t* __restrict__ additional_count,
    float*   __restrict__ profit,
    bool*    __restrict__ trade_open,
    float*   __restrict__ trade_realized_pnl,
    int32_t* __restrict__ trade_total_count,
    int32_t* __restrict__ trade_win_count,
    float*   __restrict__ trade_holding_sum,
    float*   __restrict__ trade_holding_avg,
    int32_t* __restrict__ curr_win_streak,
    int32_t* __restrict__ curr_loss_streak,
    int32_t* __restrict__ max_win_streak,
    int32_t* __restrict__ max_loss_streak,
    const int64_t* __restrict__ holding_period,
    // lot ledger (flattened)
    bool*    __restrict__ lot_open,
    int*     __restrict__ lot_node_idx,
    int8_t*  __restrict__ lot_side,
    float*   __restrict__ lot_entry,
    float*   __restrict__ lot_exposure,
    int*     __restrict__ lot_open_step,
    // node accumulators
    int32_t* __restrict__ node_apply_count,
    int32_t* __restrict__ node_noop_count,
    int32_t* __restrict__ node_win_count,
    float*   __restrict__ node_realized_pnl_sum,
    float*   __restrict__ node_exposure_sum,
    // dims
    int B, int L, int N, int current_step,
    // scalars
    float buy_exec_price,
    float sell_exec_price,
    float taker_fee_rate,
    float min_entry_ratio,
    long long min_leverage,
    long long max_leverage,
    float min_additional_enter_ratio,
    long long max_additional_entries
){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    const float a0 = actions[i * 4 + 0];
    const float a1 = actions[i * 4 + 1];
    const float a2 = actions[i * 4 + 2];
    const int atype = static_cast<int>(llroundf(a0));
    const int src_node = action_node_idx[i];
    if (atype == ACTION_NOT_FOUND || src_node < 0 || src_node >= N){
        return;
    }

    // Row pointers for lot and node accumulators
    bool*  lot_open_row = lot_open + (size_t)i * L;
    int*   lot_node_row = lot_node_idx + (size_t)i * L;
    int8_t* lot_side_row = lot_side + (size_t)i * L;
    float* lot_entry_row = lot_entry + (size_t)i * L;
    float* lot_expo_row  = lot_exposure + (size_t)i * L;
    int*   lot_openstep_row = lot_open_step + (size_t)i * L;

    int32_t* apply_row = node_apply_count + (size_t)i * N;
    int32_t* noop_row  = node_noop_count  + (size_t)i * N;
    int32_t* win_row   = node_win_count   + (size_t)i * N;
    float*   pnl_row   = node_realized_pnl_sum + (size_t)i * N;
    float*   exp_row   = node_exposure_sum     + (size_t)i * N;

    const int64_t pos = pos_list[i];
    auto fee_apply = [&](float margin, float levf) {
        profit[i] -= margin * levf * taker_fee_rate;
    };

    // Helper to realize a close amount on a single lot and update node accums
    auto realize_lot = [&](int k, float close_margin, float exec_price){
        if (!lot_open_row[k] || close_margin <= 0.0f) return 0;
        const int nidx = lot_node_row[k];
        const float entry = lot_entry_row[k];
        const float levf = static_cast<float>(leverage_ratio[i]);
        const float side = (lot_side_row[k] == 1) ? 1.0f : -1.0f; // 1: long, -1: short
        float pnl = ((side > 0 ? (exec_price - entry) : (entry - exec_price)) / (entry + 1e-9f)) * levf * close_margin;
        const float fee = close_margin * levf * taker_fee_rate;
        const float realized = pnl - fee;
        profit[i] += realized;
        trade_realized_pnl[i] += realized;
        pnl_row[nidx] += realized;
        // close portion
        lot_expo_row[k] = fmaxf(0.0f, lot_expo_row[k] - close_margin);
        if (lot_expo_row[k] <= min_entry_ratio) {
            lot_open_row[k] = false;
        }
        if (realized > 0.0f) win_row[nidx] += 1;
        return 1;
    };

    // OPEN: NEW LONG/SHORT
    if ((atype == ACTION_NEW_LONG || atype == ACTION_NEW_SHORT) && pos == 0){
        float ratio = fclamp(a1, min_entry_ratio, 1.0f);
        long long lev = iclamp(static_cast<long long>(llroundf(a2)), min_leverage, max_leverage);
        // apply and attribute entry fee to source node
        float fee = ratio * static_cast<float>(lev) * taker_fee_rate;
        profit[i] -= fee;
        pos_list[i] = (atype == ACTION_NEW_LONG) ? 2 : 1;
        price_list[i] = (atype == ACTION_NEW_LONG) ? buy_exec_price : sell_exec_price;
        leverage_ratio[i] = lev;
        enter_ratio[i] = ratio;
        additional_count[i] = 0;
        trade_open[i] = true;
        trade_realized_pnl[i] = 0.0f;
        // lot record
        int slot = find_free_lot(lot_open_row, L);
        if (slot >= 0){
            lot_open_row[slot] = true;
            lot_node_row[slot] = src_node;
            lot_side_row[slot] = (atype == ACTION_NEW_LONG) ? 1 : -1;
            lot_entry_row[slot] = (atype == ACTION_NEW_LONG) ? buy_exec_price : sell_exec_price;
            lot_expo_row[slot]  = ratio;
            lot_openstep_row[slot] = current_step;
            exp_row[src_node] += ratio;
            // attribute entry fee as negative PnL to opening node
            pnl_row[src_node] -= fee;
            apply_row[src_node] += 1;
        } else {
            noop_row[src_node] += 1; // no slot available
        }
        return;
    }

    // CLOSE ALL
    if (atype == ACTION_CLOSE_ALL && pos != 0){
        // realize all open lots at proper side price
        const float exec_price = (pos == 2) ? sell_exec_price : buy_exec_price;
        // Count trade
        trade_total_count[i] += 1;
        // For each lot
        for (int k = 0; k < L; ++k){
            if (!lot_open_row[k]) continue;
            const int nidx = lot_node_row[k];
            const float cm = lot_expo_row[k];
            if (cm <= 0.0f) { lot_open_row[k] = false; continue; }
            // Per-lot realize
            const float entry = lot_entry_row[k];
            const float levf = static_cast<float>(leverage_ratio[i]);
            const float side = (lot_side_row[k] == 1) ? 1.0f : -1.0f;
            float pnl = ((side > 0 ? (exec_price - entry) : (entry - exec_price)) / (entry + 1e-9f)) * levf * cm;
            const float fee = cm * levf * taker_fee_rate;
            const float realized = pnl - fee;
            profit[i] += realized;
            trade_realized_pnl[i] += realized;
            pnl_row[nidx] += realized;
            if (realized > 0.0f) win_row[nidx] += 1;
            lot_open_row[k] = false;
            lot_expo_row[k] = 0.0f;
        }
        if (trade_realized_pnl[i] > 0.0f) trade_win_count[i] += 1;
        on_trade_close(i, trade_realized_pnl[i], curr_win_streak, curr_loss_streak,
                       max_win_streak, max_loss_streak, trade_holding_sum,
                       holding_period, trade_total_count, trade_holding_avg);
        trade_open[i] = false;
        trade_realized_pnl[i] = 0.0f;
        pos_list[i] = 0; price_list[i] = -1.0f; leverage_ratio[i] = -1; enter_ratio[i] = -1.0f; additional_count[i] = 0;
        apply_row[src_node] += 1;
        return;
    }

    // CLOSE PARTIAL
    if (atype == ACTION_CLOSE_PARTIAL && pos != 0){
        const float ratio = fmaxf(0.0f, fminf(1.0f, a1));
        const float er_cur = enter_ratio[i];
        if (er_cur <= 0.0f || ratio <= 0.0f){
            noop_row[src_node] += 1; return;
        }
        const float total_close = er_cur * ratio;
        float remaining_to_close = total_close;
        const float exec_price = (pos == 2) ? sell_exec_price : buy_exec_price;
        // distribute pro-rata by exposure
        float sum_expo = 0.0f; for (int k=0;k<L;++k){ if (lot_open_row[k]) sum_expo += lot_expo_row[k]; }
        if (sum_expo <= 0.0f){ noop_row[src_node] += 1; return; }
        for (int k = 0; k < L && remaining_to_close > 1e-9f; ++k){
            if (!lot_open_row[k]) continue;
            float share = lot_expo_row[k] / (sum_expo + 1e-9f);
            float close_k = fminf(lot_expo_row[k], total_close * share);
            if (close_k <= 0.0f) continue;
            // realize portion
            const int nidx = lot_node_row[k];
            const float entry = lot_entry_row[k];
            const float levf = static_cast<float>(leverage_ratio[i]);
            const float side = (lot_side_row[k] == 1) ? 1.0f : -1.0f;
            float pnl = ((side > 0 ? (exec_price - entry) : (entry - exec_price)) / (entry + 1e-9f)) * levf * close_k;
            const float fee = close_k * levf * taker_fee_rate;
            const float realized = pnl - fee;
            profit[i] += realized;
            trade_realized_pnl[i] += realized; // accumulate
            pnl_row[nidx] += realized;
            if (realized > 0.0f) win_row[nidx] += 1;
            lot_expo_row[k] = fmaxf(0.0f, lot_expo_row[k] - close_k);
            if (lot_expo_row[k] <= min_entry_ratio) lot_open_row[k] = false;
            remaining_to_close -= close_k;
        }
        // Update aggregated er
        enter_ratio[i] = fmaxf(0.0f, er_cur - total_close);
        // Decrement additional count but not below zero
        additional_count[i] = (additional_count[i] > 0) ? additional_count[i] - 1 : 0;
        apply_row[src_node] += 1;
        return;
    }

    // ADD POSITION
    if (atype == ACTION_ADD_POSITION && pos != 0 && additional_count[i] < max_additional_entries){
        const float er = enter_ratio[i];
        const float remaining = fmaxf(0.0f, 1.0f - er);
        if (remaining >= min_additional_enter_ratio){
            const float desired = fmaxf(a1, min_additional_enter_ratio);
            const float add = fminf(desired, remaining);
            if (add > 0.0f){
                const float levf = static_cast<float>(leverage_ratio[i]);
                // apply and attribute entry fee for add
                float fee = add * levf * taker_fee_rate;
                profit[i] -= fee;
                // update aggregated avg price
                const float old_price = price_list[i];
                const float new_avg = (old_price * er + ((pos == 2) ? buy_exec_price : sell_exec_price) * add) / (er + add + 1e-9f);
                price_list[i] = new_avg;
                enter_ratio[i] = er + add;
                additional_count[i] += 1;
                // lot record
                int slot = find_free_lot(lot_open_row, L);
                if (slot >= 0){
                    lot_open_row[slot] = true;
                    lot_node_row[slot] = src_node;
                    lot_side_row[slot] = (pos == 2) ? 1 : -1;
                    lot_entry_row[slot] = (pos == 2) ? buy_exec_price : sell_exec_price;
                    lot_expo_row[slot]  = add;
                    lot_openstep_row[slot] = current_step;
                    exp_row[src_node] += add;
                    // attribute add entry fee as negative PnL to node
                    pnl_row[src_node] -= fee;
                    apply_row[src_node] += 1;
                } else {
                    noop_row[src_node] += 1;
                }
                return;
            }
        }
        noop_row[src_node] += 1; return;
    }

    // FLIP POSITION
    if (atype == ACTION_FLIP_POSITION && pos != 0){
        // close all first at proper price
        const float exec_price = (pos == 2) ? sell_exec_price : buy_exec_price;
        trade_total_count[i] += 1;
        for (int k = 0; k < L; ++k){
            if (!lot_open_row[k]) continue;
            const int nidx = lot_node_row[k];
            const float cm = lot_expo_row[k];
            if (cm <= 0.0f) { lot_open_row[k] = false; continue; }
            const float entry = lot_entry_row[k];
            const float levf = static_cast<float>(leverage_ratio[i]);
            const float side = (lot_side_row[k] == 1) ? 1.0f : -1.0f;
            float pnl = ((side > 0 ? (exec_price - entry) : (entry - exec_price)) / (entry + 1e-9f)) * levf * cm;
            const float fee = cm * levf * taker_fee_rate;
            const float realized = pnl - fee;
            profit[i] += realized;
            trade_realized_pnl[i] += realized;
            pnl_row[nidx] += realized;
            if (realized > 0.0f) win_row[nidx] += 1;
            lot_open_row[k] = false;
            lot_expo_row[k] = 0.0f;
        }
        if (trade_realized_pnl[i] > 0.0f) trade_win_count[i] += 1;
        on_trade_close(i, trade_realized_pnl[i], curr_win_streak, curr_loss_streak,
                       max_win_streak, max_loss_streak, trade_holding_sum,
                       holding_period, trade_total_count, trade_holding_avg);
        trade_open[i] = false;
        trade_realized_pnl[i] = 0.0f;

        // open new opposite lot
        float ratio = fclamp(a1, min_entry_ratio, 1.0f);
        long long lev = iclamp(static_cast<long long>(llroundf(a2)), min_leverage, max_leverage);
        // apply and attribute entry fee for the new flipped position
        float fee_open = ratio * static_cast<float>(lev) * taker_fee_rate;
        profit[i] -= fee_open;
        const int64_t new_pos = (pos == 2) ? 1 : 2;
        pos_list[i] = new_pos;
        price_list[i] = (new_pos == 2) ? buy_exec_price : sell_exec_price;
        enter_ratio[i] = ratio;
        leverage_ratio[i] = lev;
        additional_count[i] = 0;
        trade_open[i] = true;
        trade_realized_pnl[i] = 0.0f;

        int slot = find_free_lot(lot_open_row, L);
        if (slot >= 0){
            lot_open_row[slot] = true;
            lot_node_row[slot] = src_node;
            lot_side_row[slot] = (new_pos == 2) ? 1 : -1;
            lot_entry_row[slot] = (new_pos == 2) ? buy_exec_price : sell_exec_price;
            lot_expo_row[slot]  = ratio;
            lot_openstep_row[slot] = current_step;
            exp_row[src_node] += ratio;
            // attribute flip-opening fee to node
            pnl_row[src_node] -= fee_open;
            apply_row[src_node] += 1;
        } else {
            noop_row[src_node] += 1;
        }
        return;
    }

    // If reached here, action had no effect
    noop_row[src_node] += 1;
}

void execute_actions_with_attrib_batch_cuda(
    torch::Tensor actions,
    torch::Tensor action_node_idx,
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
    torch::Tensor lot_open,
    torch::Tensor lot_node_idx,
    torch::Tensor lot_side,
    torch::Tensor lot_entry,
    torch::Tensor lot_exposure,
    torch::Tensor lot_open_step,
    torch::Tensor node_attempt_count, // unused here (attempts updated in Python)
    torch::Tensor node_apply_count,
    torch::Tensor node_noop_count,
    torch::Tensor node_win_count,
    torch::Tensor node_realized_pnl_sum,
    torch::Tensor node_exposure_sum,
    int current_step,
    float buy_exec_price,
    float sell_exec_price,
    float taker_fee_rate,
    float min_entry_ratio,
    int64_t min_leverage,
    int64_t max_leverage,
    float min_additional_enter_ratio,
    int64_t max_additional_entries)
{
    TORCH_CHECK(actions.is_cuda() && actions.is_contiguous() && actions.dim()==2 && actions.size(1)>=3, "actions must be (B,>=3) CUDA contiguous");
    TORCH_CHECK(action_node_idx.is_cuda() && action_node_idx.dtype()==torch::kInt32 && action_node_idx.dim()==1, "action_node_idx must be (B,) int32 CUDA");
    auto check = [](const torch::Tensor& t, const char* n){ TORCH_CHECK(t.is_cuda() && t.is_contiguous(), n, " must be CUDA contiguous"); };
    check(pos_list, "pos_list"); check(price_list, "price_list"); check(leverage_ratio, "leverage_ratio"); check(enter_ratio, "enter_ratio");
    check(additional_count, "additional_count"); check(profit, "profit"); check(trade_open, "trade_open"); check(trade_realized_pnl, "trade_realized_pnl");
    check(trade_total_count, "trade_total_count"); check(trade_win_count, "trade_win_count"); check(trade_holding_sum, "trade_holding_sum"); check(trade_holding_avg, "trade_holding_avg");
    check(curr_win_streak, "curr_win_streak"); check(curr_loss_streak, "curr_loss_streak"); check(max_win_streak, "max_win_streak"); check(max_loss_streak, "max_loss_streak"); check(holding_period, "holding_period");
    check(lot_open, "lot_open"); check(lot_node_idx, "lot_node_idx"); check(lot_side, "lot_side"); check(lot_entry, "lot_entry"); check(lot_exposure, "lot_exposure"); check(lot_open_step, "lot_open_step");
    check(node_apply_count, "node_apply_count"); check(node_noop_count, "node_noop_count"); check(node_win_count, "node_win_count"); check(node_realized_pnl_sum, "node_realized_pnl_sum"); check(node_exposure_sum, "node_exposure_sum");

    const int B = actions.size(0);
    if (B == 0) return;
    TORCH_CHECK(action_node_idx.size(0)==B, "action_node_idx must match B");
    const int L = lot_open.size(1);
    const int N = node_apply_count.size(1);

    const int threads = 256;
    const int blocks = (B + threads - 1) / threads;
    execute_actions_with_attrib_kernel<<<blocks, threads>>>(
        actions.data_ptr<float>(),
        action_node_idx.data_ptr<int>(),
        pos_list.data_ptr<int64_t>(),
        price_list.data_ptr<float>(),
        leverage_ratio.data_ptr<int64_t>(),
        enter_ratio.data_ptr<float>(),
        additional_count.data_ptr<int64_t>(),
        profit.data_ptr<float>(),
        trade_open.data_ptr<bool>(),
        trade_realized_pnl.data_ptr<float>(),
        trade_total_count.data_ptr<int32_t>(),
        trade_win_count.data_ptr<int32_t>(),
        trade_holding_sum.data_ptr<float>(),
        trade_holding_avg.data_ptr<float>(),
        curr_win_streak.data_ptr<int32_t>(),
        curr_loss_streak.data_ptr<int32_t>(),
        max_win_streak.data_ptr<int32_t>(),
        max_loss_streak.data_ptr<int32_t>(),
        holding_period.data_ptr<int64_t>(),
        lot_open.data_ptr<bool>(),
        lot_node_idx.data_ptr<int>(),
        lot_side.data_ptr<int8_t>(),
        lot_entry.data_ptr<float>(),
        lot_exposure.data_ptr<float>(),
        lot_open_step.data_ptr<int>(),
        node_apply_count.data_ptr<int32_t>(),
        node_noop_count.data_ptr<int32_t>(),
        node_win_count.data_ptr<int32_t>(),
        node_realized_pnl_sum.data_ptr<float>(),
        node_exposure_sum.data_ptr<float>(),
        B, L, N, current_step,
        buy_exec_price,
        sell_exec_price,
        taker_fee_rate,
        min_entry_ratio,
        (long long)min_leverage,
        (long long)max_leverage,
        min_additional_enter_ratio,
        (long long)max_additional_entries
    );
}

// ===============================================================
// Accumulate realized PnL and stats (thread-per-individual)
// ===============================================================

__global__ void accumulate_realized_stats_kernel(
    const float* __restrict__ profit,
    float* __restrict__ realized_cum,
    float* __restrict__ sum_returns,
    int32_t* __restrict__ count_returns,
    float* __restrict__ sum_sq_returns,
    float* __restrict__ sum_neg,
    int32_t* __restrict__ count_neg,
    float* __restrict__ sum_sq_neg,
    float* __restrict__ total_profit_agg,
    float* __restrict__ total_loss_agg,
    int32_t* __restrict__ count_wins,
    float* __restrict__ compound_value,
    int B)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    const float p = profit[i];
    if (p == 0.0f) return; // exact zero: mirrors Python mask

    realized_cum[i] += p;
    sum_returns[i] += p;
    count_returns[i] += 1;
    const float pp = p * p;
    sum_sq_returns[i] += pp;

    if (p < 0.0f) {
        sum_neg[i] += p;
        count_neg[i] += 1;
        sum_sq_neg[i] += pp;
        total_loss_agg[i] += -p;
    } else { // p > 0.0f
        total_profit_agg[i] += p;
        count_wins[i] += 1;
    }

    compound_value[i] *= (1.0f + p);
}

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
    torch::Tensor compound_value)
{
    auto check_cuda_contig = [](const torch::Tensor& t, const char* name){
        TORCH_CHECK(t.is_cuda(), name, " must be CUDA tensor");
        TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    };

    check_cuda_contig(profit, "profit");
    check_cuda_contig(realized_cum, "realized_cum");
    check_cuda_contig(sum_returns, "sum_returns");
    check_cuda_contig(count_returns, "count_returns");
    check_cuda_contig(sum_sq_returns, "sum_sq_returns");
    check_cuda_contig(sum_neg, "sum_neg");
    check_cuda_contig(count_neg, "count_neg");
    check_cuda_contig(sum_sq_neg, "sum_sq_neg");
    check_cuda_contig(total_profit_agg, "total_profit_agg");
    check_cuda_contig(total_loss_agg, "total_loss_agg");
    check_cuda_contig(count_wins, "count_wins");
    check_cuda_contig(compound_value, "compound_value");

    const int B = static_cast<int>(profit.size(0));
    if (B == 0) return;

    const int threads = 256;
    const int blocks = (B + threads - 1) / threads;

    accumulate_realized_stats_kernel<<<blocks, threads>>>(
        profit.data_ptr<float>(),
        realized_cum.data_ptr<float>(),
        sum_returns.data_ptr<float>(),
        count_returns.data_ptr<int32_t>(),
        sum_sq_returns.data_ptr<float>(),
        sum_neg.data_ptr<float>(),
        count_neg.data_ptr<int32_t>(),
        sum_sq_neg.data_ptr<float>(),
        total_profit_agg.data_ptr<float>(),
        total_loss_agg.data_ptr<float>(),
        count_wins.data_ptr<int32_t>(),
        compound_value.data_ptr<float>(),
        B
    );
}

// ===============================================================
// Liquidation check (thread-per-individual)
// ===============================================================

__global__ void check_liquidation_kernel(
    int64_t* __restrict__ pos_list,
    float* __restrict__ price_list,
    int64_t* __restrict__ leverage_ratio,
    float* __restrict__ enter_ratio,
    int64_t* __restrict__ additional_count,
    int64_t* __restrict__ holding_period,
    float* __restrict__ profit,
    bool* __restrict__ trade_open,
    float* __restrict__ trade_realized_pnl,
    int32_t* __restrict__ trade_total_count,
    int32_t* __restrict__ trade_win_count,
    float* __restrict__ trade_holding_sum,
    float* __restrict__ trade_holding_avg,
    int32_t* __restrict__ curr_win_streak,
    int32_t* __restrict__ curr_loss_streak,
    int32_t* __restrict__ max_win_streak,
    int32_t* __restrict__ max_loss_streak,
    int B,
    float mark_low,
    float mark_high,
    float maintenance_margin_rate,
    float liq_clearance_fee_rate)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    const int64_t pos = pos_list[i];
    if (pos == 0) return; // no position

    const float entry = price_list[i];
    const float levf = static_cast<float>(leverage_ratio[i]);
    const float er = enter_ratio[i];

    // Guard: invalid placeholders should not trigger work
    if (er <= 0.0f || levf <= 0.0f || entry <= 0.0f) return;

    // Compute margin balance vs maintenance threshold
    const float denom = entry + 1e-9f;
    float pnl_pct = 0.0f;
    if (pos == 2) { // LONG -> risk at bar low
        pnl_pct = (mark_low - entry) / denom;
    } else { // SHORT -> risk at bar high
        pnl_pct = (entry - mark_high) / denom;
    }
    const float margin_balance = er * (1.0f + pnl_pct * levf);
    const float maintenance_margin = (er * levf) * maintenance_margin_rate;

    if (margin_balance <= maintenance_margin) {
        // Initial margin loss + liquidation clearance fee
        const float notional = er * levf;
        const float liq_fee = notional * liq_clearance_fee_rate;
        const float realized_delta = -(er + liq_fee);

        // Apply realized to per-step profit and trade accumulators
        profit[i] += realized_delta;
        trade_realized_pnl[i] += realized_delta;
        trade_total_count[i] += 1;
        if (trade_realized_pnl[i] > 0.0f) trade_win_count[i] += 1;

        // Close trade: update streaks and holding aggregates
        on_trade_close(i, trade_realized_pnl[i],
                       curr_win_streak, curr_loss_streak,
                       max_win_streak, max_loss_streak,
                       trade_holding_sum,
                       holding_period,
                       trade_total_count,
                       trade_holding_avg);

        trade_open[i] = false;
        trade_realized_pnl[i] = 0.0f;

        // Reset state
        pos_list[i] = 0;
        price_list[i] = -1.0f;
        leverage_ratio[i] = -1;
        enter_ratio[i] = -1.0f;
        additional_count[i] = 0;
        holding_period[i] = 0;
    }
}

// ===============================================================
// Liquidation check with node attribution (thread-per-individual)
// ===============================================================

__global__ void check_liquidation_with_attrib_kernel(
    int64_t* __restrict__ pos_list,
    float* __restrict__ price_list,
    int64_t* __restrict__ leverage_ratio,
    float* __restrict__ enter_ratio,
    int64_t* __restrict__ additional_count,
    int64_t* __restrict__ holding_period,
    float* __restrict__ profit,
    bool* __restrict__ trade_open,
    float* __restrict__ trade_realized_pnl,
    int32_t* __restrict__ trade_total_count,
    int32_t* __restrict__ trade_win_count,
    float* __restrict__ trade_holding_sum,
    float* __restrict__ trade_holding_avg,
    int32_t* __restrict__ curr_win_streak,
    int32_t* __restrict__ curr_loss_streak,
    int32_t* __restrict__ max_win_streak,
    int32_t* __restrict__ max_loss_streak,
    // Lot ledger
    bool* __restrict__ lot_open,
    int*  __restrict__ lot_node_idx,
    int8_t* __restrict__ lot_side,
    float* __restrict__ lot_entry,
    float* __restrict__ lot_exposure,
    int* __restrict__ lot_open_step,
    // Node accumulators
    int32_t* __restrict__ node_win_count,
    float* __restrict__ node_realized_pnl_sum,
    float* __restrict__ node_exposure_sum,
    // sizes
    int B,
    int L,
    int N,
    // marks and params
    float mark_low,
    float mark_high,
    float maintenance_margin_rate,
    float liq_clearance_fee_rate)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    const int64_t pos = pos_list[i];
    if (pos == 0) return; // no position open

    const float entry = price_list[i];
    const float levf = static_cast<float>(leverage_ratio[i]);
    const float er = enter_ratio[i];
    if (er <= 0.0f || levf <= 0.0f || entry <= 0.0f) return;

    // Compute margin balance vs maintenance threshold at conservative mark
    const float denom = entry + 1e-9f;
    float pnl_pct = 0.0f;
    if (pos == 2) {
        pnl_pct = (mark_low - entry) / denom;
    } else {
        pnl_pct = (entry - mark_high) / denom;
    }
    const float margin_balance = er * (1.0f + pnl_pct * levf);
    const float maintenance_margin = (er * levf) * maintenance_margin_rate;

    if (margin_balance <= maintenance_margin) {
        // Liquidation realized delta: initial margin + clearance fee
        const float notional = er * levf;
        const float liq_fee = notional * liq_clearance_fee_rate;
        const float realized_delta = -(er + liq_fee);

        // Attribute realized_delta across open lots by exposure share
        // Row-major access helpers
        bool* lot_open_row = lot_open + (size_t)i * (size_t)L;
        int*  lot_node_row = lot_node_idx + (size_t)i * (size_t)L;
        int8_t* lot_side_row = lot_side + (size_t)i * (size_t)L; // unused, kept for parity
        float* lot_entry_row = lot_entry + (size_t)i * (size_t)L; // unused, kept for parity
        float* lot_expo_row = lot_exposure + (size_t)i * (size_t)L;
        int* lot_openstep_row = lot_open_step + (size_t)i * (size_t)L; // unused

        float sum_expo = 0.0f;
        for (int k = 0; k < L; ++k) {
            if (lot_open_row[k]) sum_expo += lot_expo_row[k];
        }

        if (sum_expo > 0.0f) {
            for (int k = 0; k < L; ++k) {
                if (!lot_open_row[k]) continue;
                const int nidx = lot_node_row[k];
                if (nidx < 0 || nidx >= N) { lot_open_row[k] = false; lot_expo_row[k] = 0.0f; continue; }
                const float cm = lot_expo_row[k];
                const float share = cm / (sum_expo + 1e-9f);
                const float realized_k = realized_delta * share;
                node_realized_pnl_sum[(size_t)i * (size_t)N + (size_t)nidx] += realized_k;
                if (realized_k > 0.0f) node_win_count[(size_t)i * (size_t)N + (size_t)nidx] += 1;
                node_exposure_sum[(size_t)i * (size_t)N + (size_t)nidx] += cm;
                // Close lot
                lot_open_row[k] = false;
                lot_expo_row[k] = 0.0f;
            }
        } else {
            // No open lots recorded; nothing to attribute
        }

        // Apply realized to per-step profit and trade accumulators
        profit[i] += realized_delta;
        trade_realized_pnl[i] += realized_delta;
        trade_total_count[i] += 1;
        if (trade_realized_pnl[i] > 0.0f) trade_win_count[i] += 1;

        // Close trade: update streaks and holding aggregates
        on_trade_close(i, trade_realized_pnl[i],
                       curr_win_streak, curr_loss_streak,
                       max_win_streak, max_loss_streak,
                       trade_holding_sum,
                       holding_period,
                       trade_total_count,
                       trade_holding_avg);

        trade_open[i] = false;
        trade_realized_pnl[i] = 0.0f;

        // Reset aggregate state
        pos_list[i] = 0;
        price_list[i] = -1.0f;
        leverage_ratio[i] = -1;
        enter_ratio[i] = -1.0f;
        additional_count[i] = 0;
        holding_period[i] = 0;
    }
}

void check_liquidation_with_attrib_cuda(
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
    torch::Tensor lot_open,
    torch::Tensor lot_node_idx,
    torch::Tensor lot_side,
    torch::Tensor lot_entry,
    torch::Tensor lot_exposure,
    torch::Tensor lot_open_step,
    torch::Tensor node_win_count,
    torch::Tensor node_realized_pnl_sum,
    torch::Tensor node_exposure_sum,
    float mark_low,
    float mark_high,
    float maintenance_margin_rate,
    float liq_clearance_fee_rate)
{
    auto check_cuda_contig = [](const torch::Tensor& t, const char* name){
        TORCH_CHECK(t.is_cuda(), name, " must be CUDA tensor");
        TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    };

    check_cuda_contig(pos_list, "pos_list");
    check_cuda_contig(price_list, "price_list");
    check_cuda_contig(leverage_ratio, "leverage_ratio");
    check_cuda_contig(enter_ratio, "enter_ratio");
    check_cuda_contig(additional_count, "additional_count");
    check_cuda_contig(holding_period, "holding_period");
    check_cuda_contig(profit, "profit");
    check_cuda_contig(trade_open, "trade_open");
    check_cuda_contig(trade_realized_pnl, "trade_realized_pnl");
    check_cuda_contig(trade_total_count, "trade_total_count");
    check_cuda_contig(trade_win_count, "trade_win_count");
    check_cuda_contig(trade_holding_sum, "trade_holding_sum");
    check_cuda_contig(trade_holding_avg, "trade_holding_avg");
    check_cuda_contig(curr_win_streak, "curr_win_streak");
    check_cuda_contig(curr_loss_streak, "curr_loss_streak");
    check_cuda_contig(max_win_streak, "max_win_streak");
    check_cuda_contig(max_loss_streak, "max_loss_streak");

    check_cuda_contig(lot_open, "lot_open");
    check_cuda_contig(lot_node_idx, "lot_node_idx");
    check_cuda_contig(lot_side, "lot_side");
    check_cuda_contig(lot_entry, "lot_entry");
    check_cuda_contig(lot_exposure, "lot_exposure");
    check_cuda_contig(lot_open_step, "lot_open_step");

    check_cuda_contig(node_win_count, "node_win_count");
    check_cuda_contig(node_realized_pnl_sum, "node_realized_pnl_sum");
    check_cuda_contig(node_exposure_sum, "node_exposure_sum");

    const int B = static_cast<int>(pos_list.size(0));
    if (B == 0) return;
    const int L = static_cast<int>(lot_open.size(1));
    const int N = static_cast<int>(node_win_count.size(1));

    const int threads = 256;
    const int blocks = (B + threads - 1) / threads;
    check_liquidation_with_attrib_kernel<<<blocks, threads>>>(
        pos_list.data_ptr<int64_t>(),
        price_list.data_ptr<float>(),
        leverage_ratio.data_ptr<int64_t>(),
        enter_ratio.data_ptr<float>(),
        additional_count.data_ptr<int64_t>(),
        holding_period.data_ptr<int64_t>(),
        profit.data_ptr<float>(),
        trade_open.data_ptr<bool>(),
        trade_realized_pnl.data_ptr<float>(),
        trade_total_count.data_ptr<int32_t>(),
        trade_win_count.data_ptr<int32_t>(),
        trade_holding_sum.data_ptr<float>(),
        trade_holding_avg.data_ptr<float>(),
        curr_win_streak.data_ptr<int32_t>(),
        curr_loss_streak.data_ptr<int32_t>(),
        max_win_streak.data_ptr<int32_t>(),
        max_loss_streak.data_ptr<int32_t>(),
        lot_open.data_ptr<bool>(),
        lot_node_idx.data_ptr<int>(),
        lot_side.data_ptr<int8_t>(),
        lot_entry.data_ptr<float>(),
        lot_exposure.data_ptr<float>(),
        lot_open_step.data_ptr<int>(),
        node_win_count.data_ptr<int32_t>(),
        node_realized_pnl_sum.data_ptr<float>(),
        node_exposure_sum.data_ptr<float>(),
        B, L, N,
        mark_low,
        mark_high,
        maintenance_margin_rate,
        liq_clearance_fee_rate
    );
}

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
    float liq_clearance_fee_rate)
{
    auto check_cuda_contig = [](const torch::Tensor& t, const char* name){
        TORCH_CHECK(t.is_cuda(), name, " must be CUDA tensor");
        TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    };

    check_cuda_contig(pos_list, "pos_list");
    check_cuda_contig(price_list, "price_list");
    check_cuda_contig(leverage_ratio, "leverage_ratio");
    check_cuda_contig(enter_ratio, "enter_ratio");
    check_cuda_contig(additional_count, "additional_count");
    check_cuda_contig(holding_period, "holding_period");
    check_cuda_contig(profit, "profit");
    check_cuda_contig(trade_open, "trade_open");
    check_cuda_contig(trade_realized_pnl, "trade_realized_pnl");
    check_cuda_contig(trade_total_count, "trade_total_count");
    check_cuda_contig(trade_win_count, "trade_win_count");
    check_cuda_contig(trade_holding_sum, "trade_holding_sum");
    check_cuda_contig(trade_holding_avg, "trade_holding_avg");
    check_cuda_contig(curr_win_streak, "curr_win_streak");
    check_cuda_contig(curr_loss_streak, "curr_loss_streak");
    check_cuda_contig(max_win_streak, "max_win_streak");
    check_cuda_contig(max_loss_streak, "max_loss_streak");

    const int B = static_cast<int>(pos_list.size(0));
    if (B == 0) return;
    TORCH_CHECK(price_list.size(0) == B, "shape mismatch");

    const int threads = 256;
    const int blocks = (B + threads - 1) / threads;

    check_liquidation_kernel<<<blocks, threads>>>(
        pos_list.data_ptr<int64_t>(),
        price_list.data_ptr<float>(),
        leverage_ratio.data_ptr<int64_t>(),
        enter_ratio.data_ptr<float>(),
        additional_count.data_ptr<int64_t>(),
        holding_period.data_ptr<int64_t>(),
        profit.data_ptr<float>(),
        trade_open.data_ptr<bool>(),
        trade_realized_pnl.data_ptr<float>(),
        trade_total_count.data_ptr<int32_t>(),
        trade_win_count.data_ptr<int32_t>(),
        trade_holding_sum.data_ptr<float>(),
        trade_holding_avg.data_ptr<float>(),
        curr_win_streak.data_ptr<int32_t>(),
        curr_loss_streak.data_ptr<int32_t>(),
        max_win_streak.data_ptr<int32_t>(),
        max_loss_streak.data_ptr<int32_t>(),
        B,
        mark_low,
        mark_high,
        maintenance_margin_rate,
        liq_clearance_fee_rate
    );
}

// ===============================================================
// Compute equity (realized + unrealized) (thread-per-individual)
// ===============================================================

__global__ void compute_equity_kernel(
    const int64_t* __restrict__ pos_list,
    const float* __restrict__ price_list,
    const int64_t* __restrict__ leverage_ratio,
    const float* __restrict__ enter_ratio,
    const float* __restrict__ realized_cum,
    float mark_t,
    float equity_base,
    float* __restrict__ equity_out,
    int B)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    const int64_t pos = pos_list[i];
    const float entry = price_list[i];
    const float er = enter_ratio[i];
    const float levf = static_cast<float>(leverage_ratio[i]);

    float unreal = 0.0f;
    if (pos != 0 && entry > 0.0f && er > 0.0f && levf > 0.0f) {
        const float denom = entry + 1e-9f;
        const float side = (pos == 2) ? 1.0f : -1.0f; // 2: long, 1: short
        const float pnl_pct = side * (mark_t - entry) / denom;
        unreal = pnl_pct * levf * er;
    }
    equity_out[i] = equity_base + realized_cum[i] + unreal;
}

void compute_equity_batch_cuda(
    torch::Tensor pos_list,
    torch::Tensor price_list,
    torch::Tensor leverage_ratio,
    torch::Tensor enter_ratio,
    torch::Tensor realized_cum,
    float mark_t,
    float equity_base,
    torch::Tensor equity_out)
{
    auto check_cuda_contig = [](const torch::Tensor& t, const char* name){
        TORCH_CHECK(t.is_cuda(), name, " must be CUDA tensor");
        TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    };

    check_cuda_contig(pos_list, "pos_list");
    check_cuda_contig(price_list, "price_list");
    check_cuda_contig(leverage_ratio, "leverage_ratio");
    check_cuda_contig(enter_ratio, "enter_ratio");
    check_cuda_contig(realized_cum, "realized_cum");
    check_cuda_contig(equity_out, "equity_out");

    const int B = static_cast<int>(pos_list.size(0));
    if (B == 0) return;
    TORCH_CHECK(price_list.size(0) == B, "shape mismatch");

    const int threads = 256;
    const int blocks = (B + threads - 1) / threads;

    compute_equity_kernel<<<blocks, threads>>>(
        pos_list.data_ptr<int64_t>(),
        price_list.data_ptr<float>(),
        leverage_ratio.data_ptr<int64_t>(),
        enter_ratio.data_ptr<float>(),
        realized_cum.data_ptr<float>(),
        mark_t,
        equity_base,
        equity_out.data_ptr<float>(),
        B
    );
}
