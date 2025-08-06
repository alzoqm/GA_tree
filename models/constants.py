# models/constants.py (신규 파일)

# -------------------------------------
# --- 상수 정의 (프로젝트 공통 사용)
# -------------------------------------

# Tensor Column Indices
COL_NODE_TYPE = 0
COL_PARENT_IDX = 1
COL_DEPTH = 2
COL_PARAM_1 = 3
COL_PARAM_2 = 4
COL_PARAM_3 = 5
COL_PARAM_4 = 6
NODE_INFO_DIM = 7  # 총 컬럼 수

# Node Types
NODE_TYPE_UNUSED = 0
NODE_TYPE_ROOT_BRANCH = 1
NODE_TYPE_DECISION = 2
NODE_TYPE_ACTION = 3

# Root Branch Types
ROOT_BRANCH_LONG = 0
ROOT_BRANCH_HOLD = 1
ROOT_BRANCH_SHORT = 2

# Decision Node: Comparison Types
COMP_TYPE_FEAT_NUM = 0  # Feature vs Number
COMP_TYPE_FEAT_FEAT = 1 # Feature vs Feature
COMP_TYPE_FEAT_BOOL = 2 # Feature vs Boolean

# Decision Node: Operators
OP_GTE = 0  # >= (Greater Than or Equal)
OP_LTE = 1  # <= (Less Than or Equal)

# Action Node: Action Types (COL_PARAM_1에 저장)
ACTION_NOT_FOUND     = 0  # CUDA 커널에서 Action을 찾지 못했을 때의 기본값
ACTION_NEW_LONG      = 1  # 신규 롱 포지션 진입
ACTION_NEW_SHORT     = 2  # 신규 숏 포지션 진입
ACTION_CLOSE_ALL     = 3  # 현재 포지션 전체 청산
ACTION_CLOSE_PARTIAL = 4  # 현재 포지션 부분 청산 (비율 지정)
ACTION_ADD_POSITION  = 5  # 현재 포지션 추가 진입 (피라미딩)
ACTION_FLIP_POSITION = 6  # 현재 포지션 청산 후 즉시 반대 포지션 진입

# --- Helper 딕셔너리 (시각화 및 디버깅용) ---
NODE_TYPE_MAP = {
    NODE_TYPE_UNUSED: "UNUSED",
    NODE_TYPE_ROOT_BRANCH: "ROOT_BRANCH",
    NODE_TYPE_DECISION: "DECISION",
    NODE_TYPE_ACTION: "ACTION",
}

ROOT_BRANCH_MAP = {
    ROOT_BRANCH_LONG: "IF_POS_IS_LONG",
    ROOT_BRANCH_HOLD: "IF_POS_IS_HOLD",
    ROOT_BRANCH_SHORT: "IF_POS_IS_SHORT",
}

OPERATOR_MAP = {
    OP_GTE: ">=",
    OP_LTE: "<=",
}

ACTION_TYPE_MAP = {
    ACTION_NOT_FOUND: "NOT_FOUND",
    ACTION_NEW_LONG: "NEW_LONG",
    ACTION_NEW_SHORT: "NEW_SHORT",
    ACTION_CLOSE_ALL: "CLOSE_ALL",
    ACTION_CLOSE_PARTIAL: "CLOSE_PARTIAL",
    ACTION_ADD_POSITION: "ADD_POSITION",
    ACTION_FLIP_POSITION: "FLIP_POSITION",
}