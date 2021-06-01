#include "../include/core.cuh"
#include "../include/operations.cuh"

void test_init (VLMO_Operator_Descriptor_t& desc) {

    srand(0);
    // A
    float* A = (float*) malloc (sizeof (float)*desc.A_h*desc.A_w);
    for (int i=0; i<desc.A_h; i++)
        for (int j=0; j<desc.A_w; j++)
            A[i*desc.A_w+j] = (rand ()%1000-500)/100;
    desc.host_A = A;

    // B
    float* B = (float*) malloc (sizeof (float)*desc.B_h*desc.B_w);

    for (int i=0; i<desc.B_h; i++)
        for (int j=0; j<desc.B_w; j++)
            B[i*desc.B_w+j] = (rand ()%1000-500)/100;
    desc.host_B = B;

    // C
    float* C = (float*) calloc (desc.C_h*desc.C_w, sizeof (float));
    desc.host_C = C;

    size_t total_size = sizeof(float)*desc.A_h*desc.A_w + sizeof(float)*desc.B_h*desc.B_w + sizeof(float)*desc.C_h*desc.C_w;
    printf("[Mem] Host memory allocation completed..\n");
    printf("    total memory  usage : %.3f GB\n", total_size*1e-9);

}

void test_result (VLMO_Operator_Descriptor_t& desc, float* A, float*B, float* C) {

    printf("[Test] Start checking result ..\n");
    float result = 0.0f;
    for (int i=0; i<desc.C_h; i++)
        for (int j=0; j<desc.C_w; j++) {

            switch (desc.op) {
                case VLMO_Op_Element_Add:
                    result = A[i*desc.C_w+j] + B[i*desc.C_w+j];
                    break;
                case VLMO_Op_Element_Sub:
                    result = A[i*desc.C_w+j] - B[i*desc.C_w+j];
                    break;
                case VLMO_Op_Element_Mul:
                    result = A[i*desc.C_w+j] * B[i*desc.C_w+j];
                    break;
                case VLMO_Op_Element_Div:
                    if (B[i*desc.C_w+j] != 0)
                        result = A[i*desc.C_w+j] / B[i*desc.C_w+j];
                    else
                        result = 0.0f;
                    break;
            }
            
            if (C[i*desc.C_w+j] != result) {
                printf("[Test] Test failed... C[%d, %d] = %f, but %f\n", i, j, result, C[i*desc.C_w+j]);
                return ;
            }
        }

    printf("[Test] Test success!\n");
}

int main(void) {


    /****
      *** Very Large Matrices Element-wise Operation Example with a Single Device
      *** These matrices stored in memory with "row" major
      ****/

    // Define a example problem 
    size_t w = 1024*3+40;
    size_t h = 1024*2+4;
    printf("Total size of matrix: %.3f GB\n", sizeof(float)*w*h*3*1e-9);
    int device_id = 0;
    
    // Get environment
    size_t free, total;
    cudaDeviceProp prop =  VLMO_get_device_properties (device_id, &free, &total, false);


    /** Patch based operations **/
    VLMO_Operator_t op = VLMO_Op_Mat_Mul;
    printf("=======================================================\n");
    printf ("[%s]\n", VLMO_Op_Name[op].c_str());
    printf("=======================================================\n");

    // Descript a opearator
    VLMO_Operator_Descriptor_t desc;
    desc.op = op;
    desc.A_w = desc.B_w = desc.C_w = w;
    desc.A_h = desc.B_h = desc.C_h = h;
    desc.prop = prop;
    desc.mem_free_size = free;
    desc.num_threads = dim3(16, 16);
    desc.flag_unified_mem=false;

    // Initiate data for test
    test_init (desc);

    // Allocate device memory
    VLMO_malloc_device_mem (desc, true);

    // Launch matrix addtion kernel
    printf("[Func] %s start..\n", VLMO_Op_Name[op].c_str());
    VLMO_matrix_multiplication (desc, true);
    
    // Test result
    test_result(desc, desc.host_A, desc.host_B, desc.host_C);

    // Free all memory allocations
    VLMO_clear_all (desc);
    printf("=======================================================\n\n\n");

 
    return 0;
}

