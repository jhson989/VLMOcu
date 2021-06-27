#include "../include/core.cuh"
#include "../include/operations.cuh"
#include <omp.h>



void test_init (VLMO_Operator_Descriptor_t& desc) {

    srand(0);

    // A
    float* A = (float*) malloc (sizeof (float)*desc.A_h*desc.A_w);
    /*
    for (int i=0; i<desc.A_h; i++) {
        for (int j=0; j<desc.A_w; j++) {
            A[i*desc.A_w+j] = (rand ()%5);
        }
    }
    */
    desc.host_A = A;

    // B
    float* B = (float*) malloc (sizeof (float)*desc.B_h*desc.B_w);
    /*
    for (int i=0; i<desc.B_h; i++) {
        for (int j=0; j<desc.B_w; j++) {
            B[i*desc.B_w+j] = (rand ()%5);
        }
    }
    */
    desc.host_B = B;

    // C
    float* C = (float*) calloc (desc.C_h*desc.C_w, sizeof (float));
    desc.host_C = C;

    size_t total_size = sizeof(float)*desc.A_h*desc.A_w + sizeof(float)*desc.B_h*desc.B_w + sizeof(float)*desc.C_h*desc.C_w;
    printf("[Mem] Host memory allocation completed..\n");
    printf("    total memory usage : %.3f GB\n", total_size*1e-9);
}

void test_result (VLMO_Operator_Descriptor_t& desc, float* A, float* B, float* C, const bool do_test) {

    if (do_test == false) {
        printf("[TEST] Test skipped..\n");
        return;
    }

    printf("[Test] Start checking result ..\n");
    


    bool flag_exit[NUM_CPU_CORE] = {0};
    for (size_t i=0; i<desc.C_h; i++) {
        if (i%100 == 0) {
            printf("\r    Test....[%5lu/%5lu]", i, desc.C_h);
            fflush(stdout);
        }

        #pragma omp parallel for num_threads(NUM_CPU_CORE)
        for (size_t j=0; j<desc.C_w; j++) {
            int tid = omp_get_thread_num();
            float result=0.0f;
            for (size_t l=0; l<desc.B_h; l++) {
                result += A[i*desc.A_w+l]*B[l*desc.B_w+j];
            }

            if (C[i*desc.C_w+j] != result) {
                printf("\n[Test] Test failed... C[%lu, %lu] = %f, but %f\n", i, j, result, C[i*desc.C_w+j]);
                flag_exit[tid] = true;
            }
        }

        for (int tid=0; tid<NUM_CPU_CORE; tid++)
            if (flag_exit[tid] == true) {
                printf("\n[Test] Test failed...!\n");
                return;
            }
                

    }
    
    printf("\n[Test] Test success!\n");
    return;
}



int main(int argc, char** argv) {


    /****
      *** Very Large Matrix Multiplication Example with a Single Device
      *** Matrices stored in memory with "row" major
      ****/

    // Define this problem 
    bool flag_test = false;
    if (argc >= 2)
        flag_test = (bool)atoi(argv[1]);
    size_t m = 1024*50;
    size_t n = 1024*40;
    size_t k = 1024*40;

    VLMO_Operator_t op = VLMO_Op_Mat_Mul;
    int device_id = 0;
    

    // Get device information
    size_t free, total;
    cudaDeviceProp prop =  VLMO_get_device_properties (device_id, &free, &total, false);


    // Make matrix operation description
    
    printf ("VLMO op [%s]\n", VLMO_Op_Name[op].c_str());

    VLMO_Operator_Descriptor_t desc;
    desc.op = op;
    desc.A_h = desc.C_h = m;
    desc.B_w = desc.C_w = n;
    desc.A_w = desc.B_h = k;
    desc.flag_unified_mem = false;
    desc.mem_free_size = free;
    desc.num_device = 1;
    desc.prop = prop;
    desc.num_threads = dim3(16, 16);

    // Initiate data for test
    test_init (desc);

    // Allocate device memory
    VLMO_malloc_device_mem (desc, true);

    // Launch matrix addtion kernel
    printf("[Func] %s start..\n", VLMO_Op_Name[op].c_str());
    VLMO_matrix_multiplication (desc, true);
    
    // Test result
    test_result(desc, desc.host_A, desc.host_B, desc.host_C, flag_test);

    // Free all memory allocations
    VLMO_clear_all (desc);
    printf("\nEnd..\n\n");

    return 0;
}

