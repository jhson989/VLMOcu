#include "../include/core.cuh"

void test_init (VLMO_Operator_Descriptor_t& desc) {

    srand(0);

    // A
    float* A = (float*) malloc (sizeof (float)*desc.A_h*desc.A_w)
    for (int i=0; i<desc.A_h; i++)
        for (int j=0; j<desc.A_w; j++)
            A[i*desc.A_w+j] = (rand ()%1000-500)/100;
    desc.host_A = A;

    // B
    float* B = (float*) malloc (sizeof (float)*desc.B_h*desc.B_w)
    for (int i=0; i<desc.B_h; i++)
        for (int j=0; j<desc.B_w; j++)
            B[i*desc.B_w+j] = (rand ()%1000-500)/100;
    desc.host_B = B;

    // C
    float* C = (float*) malloc (sizeof (float)*desc.C_h*desc.C_w)
    for (int i=0; i<desc.C_h; i++)
        for (int j=0; j<desc.C_w; j++)
            A[i*desc.C_w+j] = (rand ()%1000-500)/100;
    desc.host_C = C;

}

int main(void) {


    /****
      *** Very Large Matrices Addition Example with a Single Device
      *** There Matrices stored in memory with "row" major
      ****/

    // Define this problem 
    size_t m = 1024;
    size_t n = 1024;
    size_t k = 1024;
    VLMO_Operator_t op = VLMO_Add;
    int device_id = 0;
    

    // Get device information
    size_t free, total;
    cudaDeviceProp prop =  VLMO_get_device_properties (device_id, &free, &total, true);


    // Make matrix operation description
    VLMO_Operator_Descriptor_t desc;
    desc.op = op;
    desc.A_w = desc.B_h = k;
    desc.B_w = desc.C_w = n;
    desc.A_h = desc.C_h = m;
    desc.flag_unified_mem = true;
    desc.mem_free_size = free;
    desc.num_device = 1;
    desc.prop = prop;

    // Initiate data for test
    test_init (desc);

    // Allocate device memory
    VLMO_malloc_device_mem (desc);

    return 0;
}

