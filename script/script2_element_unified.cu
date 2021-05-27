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
    printf("    total usage usage : %.3f GB\n", total_size*1e-9);

}

void test_result (VLMO_Operator_Descriptor_t& desc) {

    printf("[Test] Start checking result ..\n");
    float result = 0.0f;
    for (int i=0; i<desc.C_h; i++)
        for (int j=0; j<desc.C_w; j++) {

            switch (desc.op) {
                case VLMO_Op_Element_Add:
                    result = desc.device_A[i*desc.C_w+j] + desc.device_B[i*desc.C_w+j];
                    break;
                case VLMO_Op_Element_Sub:
                    result = desc.device_A[i*desc.C_w+j] - desc.device_B[i*desc.C_w+j];
                    break;
                case VLMO_Op_Element_Mul:
                    result = desc.device_A[i*desc.C_w+j] * desc.device_B[i*desc.C_w+j];
                    break;
                case VLMO_Op_Element_Div:
                    if (desc.device_B[i*desc.C_w+j] != 0)
                        result = desc.device_A[i*desc.C_w+j] / desc.device_B[i*desc.C_w+j];
                    else
                        result = 0.0f;
                    break;
            }
            
            if (desc.device_C[i*desc.C_w+j] != result) {
                printf("[Test] Test failed... C[%d, %d] = %f, but %f\n", i, j, result, desc.device_C[i*desc.C_w+j]);
                return ;
            }
        }

    printf("[Test] Test success!\n");
}

int main(void) {


    /****
      *** Very Large Matrices Addition Example with a Single Device
      *** There Matrices stored in memory with "row" major
      ****/

    // Define this problem 
    size_t w = 1024*25;
    size_t h = 1024*25;
    const int num_op = 4;
    VLMO_Operator_t list_ops[num_op] = {
        VLMO_Op_Element_Add,
        VLMO_Op_Element_Sub,
        VLMO_Op_Element_Mul,
        VLMO_Op_Element_Div
    };
    int device_id = 0;
    

    // Get device information
    size_t free, total;
    cudaDeviceProp prop =  VLMO_get_device_properties (device_id, &free, &total, true);


    // Make matrix operation description
    
    for (int i=0; i<num_op; i++) { 
        VLMO_Operator_t op = list_ops[i];
        printf ("Do operation %s\n", VLMO_Op_Name[op].c_str());

        VLMO_Operator_Descriptor_t desc;
        desc.op = op;
        desc.A_w = desc.B_w = desc.C_w = w;
        desc.A_h = desc.B_h = desc.C_h = h;
        desc.flag_unified_mem = true;
        desc.mem_free_size = free;
        desc.num_device = 1;
        desc.prop = prop;

        // Initiate data for test
        test_init (desc);

        // Allocate device memory
        VLMO_malloc_device_mem (desc, true);

        // Launch matrix addtion kernel
        printf("[Func] %s start..\n", VLMO_Op_Name[op].c_str());
        VLMO_element_operation (desc, op, true);
        
        // Test result
        test_result(desc);

        // Free all memory allocations
        VLMO_clear_all (desc);
        printf("\nEnd..\n\n");
    }

    return 0;
}

