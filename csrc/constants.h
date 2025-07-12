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

// [수정] max_depth 추가
// 이 값은 GATreePop 생성 시 사용된 max_depth와 일치해야 합니다.
// Python 측에서 동적으로 전달하는 것이 더 안전하지만,
// 컴파일 타임 상수가 필요한 경우를 위해 여기에 정의합니다.
constexpr int MAX_DEPTH = 10;

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
constexpr int COMP_TYPE_FEAT_NUM = 0;
constexpr int COMP_TYPE_FEAT_FEAT = 1;

// --- Decision Node: Operators ---
constexpr int OP_GT = 0; // >
constexpr int OP_LT = 1; // <
constexpr int OP_EQ = 2; // =

// --- Action Node: Position Types ---
constexpr int POS_TYPE_LONG = 0;
constexpr int POS_TYPE_SHORT = 1;
// [수정] 명시적인 HOLD 타입 추가
constexpr int POS_TYPE_HOLD = 2;

// [수정] 부동소수점 비교를 위한 Epsilon
constexpr float EPSILON = 1e-6f;