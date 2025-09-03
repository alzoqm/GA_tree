// csrc/constants.h
// This file defines constants that are shared across the C++ and Python code.
// It is crucial that these values are kept in sync with the corresponding
// constants in python/model.py.

#pragma once

// --- Tensor Column Indices ---
// These constants define the column indices for accessing node information
// in the tree tensor.
constexpr int COL_NODE_TYPE = 0;      // Type of the node (e.g., UNUSED, ROOT_BRANCH, DECISION, ACTION)
constexpr int COL_PARENT_IDX = 1;     // Index of the parent node
constexpr int COL_DEPTH = 2;          // Depth of the node in the tree
constexpr int COL_PARAM_1 = 3;        // Parameter 1, usage depends on the node type
constexpr int COL_PARAM_2 = 4;        // Parameter 2, usage depends on the node type
constexpr int COL_PARAM_3 = 5;        // Parameter 3, usage depends on the node type
constexpr int COL_PARAM_4 = 6;        // Parameter 4, usage depends on the node type
constexpr int NODE_INFO_DIM = 7;      // Total number of columns for node information

// --- Node Types ---
// These constants define the different types of nodes that can exist in a tree.
constexpr int NODE_TYPE_UNUSED = 0;         // The node is not currently in use
constexpr int NODE_TYPE_ROOT_BRANCH = 1;    // The root of a branch in the tree
constexpr int NODE_TYPE_DECISION = 2;       // A decision node that evaluates a condition
constexpr int NODE_TYPE_ACTION = 3;         // A leaf node that specifies an action

// --- Root Branch Types ---
// These constants define the types of root branches.
constexpr int ROOT_BRANCH_LONG = 0;         // A long position branch
constexpr int ROOT_BRANCH_HOLD = 1;         // A hold position branch
constexpr int ROOT_BRANCH_SHORT = 2;        // A short position branch

// --- Decision Node: Comparison Types ---
// These constants define the types of comparisons that a decision node can perform.
constexpr int COMP_TYPE_FEAT_NUM = 0;       // Comparison between a feature and a number
constexpr int COMP_TYPE_FEAT_FEAT = 1;      // Comparison between two features
constexpr int COMP_TYPE_FEAT_BOOL = 2;      // Comparison between a feature and a boolean value

// --- Decision Node: Operators ---
// These constants define the operators used in decision nodes.
constexpr int OP_GTE = 0; // Greater Than or Equal (>=)
constexpr int OP_LTE = 1; // Less Than or Equal (<=)

// --- Action Node: Action Types ---
// These constants define the types of actions that an action node can represent.
// These must be kept in sync with model.py.
constexpr int ACTION_NEW_LONG        = 1;   // Open a new long position
constexpr int ACTION_NEW_SHORT       = 2;   // Open a new short position
constexpr int ACTION_CLOSE_ALL       = 3;   // Close all open positions
constexpr int ACTION_CLOSE_PARTIAL   = 4;   // Partially close a position
constexpr int ACTION_ADD_POSITION    = 5;   // Increase the size of a position
constexpr int ACTION_FLIP_POSITION   = 6;   // Flip a position from long to short or vice-versa

// Default value for Action Type, indicating that no action was found.
constexpr int ACTION_NOT_FOUND = 0;