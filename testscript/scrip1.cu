#include "../include/core.cuh"


int main(void) {


    /****
      *** Very Large Matrices Addition Example with a Single Device
      *** There Matrices stored in memory with "row" major
      ****/

    // Define this problem 
    size_t lda = 1024;
    size_t ldb = 1024;
    size_t ldc = 1024;
    VLMO_Operator_t op = VLMO_Add;
    unsigned int device_id = 1;
    

    // Get device information
    size_t free, total;
    VLMO_get_device_properties (device_id, &free, &total, true);

    VLMO_Operator_Descriptor_t desc;
    desc.op = op;
    desc.A_w = desc.B_h = lda;
    desc.B_w = desc.C_w = ldb;
    desc.A_h = desc.C_h = ldc;
    desc.flag_unified_mem = true;
    desc.mem_free_size = free;


    return 0;
}

