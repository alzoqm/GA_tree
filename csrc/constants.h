// csrc/constants.h
#pragma once

// 이 파일은 python/model.py 의 상수들과 100% 일치해야 합니다.

// --- Tensor Column Indices ---
constexpr int COL_NODE_TYPE = 0;
constexpr int COL_PARENT_IDX = 1;
constexpr int COL_DEPTH = 2;
constexpr int COL_PARAM_1 = 3;
constexpr int COL_PARAM_2 = 4;
constexpr int COL_PARAM_3 = 5;
constexpr int COL_PARAM_4 = 6;
constexpr int NODE_INFO_DIM = 7;

// --- Node Types ---
constexpr int NODE_TYPE_UNUSED = 0;
constexpr int NODE_TYPE_ROOT_BRANCH = 1;
constexpr int NODE_TYPE_DECISION = 2;
constexpr int NODE_TYPE_ACTION = 3;

// --- Root Branch Types ---
constexpr int ROOT_BRANCH_LONG = 0;
constexpr int ROOT_BRANCH_HOLD = 1;
constexpr int ROOT_BRANCH_SHORT = 2;

// --- Decision Node: Comparison Types ---
constexpr int COMP_TYPE_FEAT_NUM = 0;  // Feature vs Number
constexpr int COMP_TYPE_FEAT_FEAT = 1; // Feature vs Feature
constexpr int COMP_TYPE_FEAT_BOOL = 2; // Feature vs Boolean

// --- Decision Node: Operators ---
constexpr int OP_GTE = 0; // >= (Greater Than or Equal)
constexpr int OP_LTE = 1; // <= (Less Than or Equal)

// --- Action Node: Position Types ---
constexpr int POS_TYPE_LONG = 0;
constexpr int POS_TYPE_SHORT = 1;
// 기본값으로 사용될 HOLD. Python에서는 POS_TYPE_MAP에만 존재.
// CUDA 결과값은 숫자이므로 명시적인 상수가 있는 것이 안전.
constexpr int ACTION_DEFAULT_HOLD = -1; // 임의의 값, Python에서는 딕셔너리로 처리됨