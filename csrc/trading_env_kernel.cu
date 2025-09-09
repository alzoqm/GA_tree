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
