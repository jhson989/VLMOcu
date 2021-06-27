
#ifndef __VLMO_TYPE__
#define __VLMO_TYPE__

/***
  ** Supported Matrix Operations
  ***/
typedef enum {

    // Element-wise operation
    VLMO_OP_ELEMENT_ADD=0,
    VLMO_OP_ELEMENT_SUB,
    VLMO_OP_ELEMENT_MUL,
    VLMO_OP_ELEMENT_DIV,

    // Matrix multiplcation operation
    VLMO_OP_MAT_MUL,

    // Single matrix operation
    VLMO_OP_TRANSPOSE,

    // Default operation
    VLMO_OP_NO
} vlmoOperator_t;

// list of Operation names (
const std::string VLMO_OP_NAME[] = {

    "VLMO_OP_ELEMENT_ADD",
    "VLMO_OP_ELEMENT_SUB",
    "VLMO_OP_ELEMENT_MUL",
    "VLMO_OP_ELEMENT_DIV",
    "VLMO_OP_MAT_MUL",
    "VLMO_OP_TRANSPOSE",
    "VLMO_OP_NO"

};


#endif