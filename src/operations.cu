
#include "../include/operations.cuh"
#include "../include/kernel.cuh"
#include "../include/utils.cuh"








/******************************************************
  *****************************************************
  * Functions for element-wise opearations
  *****************************************************
  *******************************************************/

void VLMO_element_operation (VLMO_Operator_Descriptor_t& desc, VLMO_Operator_t op=VLMO_Op_No, const bool measure=false) {

    // Performance measurement
    cudaEvent_t event_start, event_stop;
    if (measure == true) {
        VLMO_record_start (event_start, event_stop);
    }

    if (op == VLMO_Op_No) 
        op = desc.op;


    switch (op) {
        case VLMO_Op_Element_Add:
            VLMO_element_addition (desc);
            break;
        case VLMO_Op_Element_Sub:
            VLMO_element_subtraction (desc);
            break;
        case VLMO_Op_Element_Mul:
            VLMO_element_multiplication (desc);
            break;
        case VLMO_Op_Element_Div:
            VLMO_element_division (desc);
            break;
    }

    // Performance measurement
    if (measure == true) {
        VLMO_record_end (event_start, event_stop);
    }

}


/** Element-wise addtion operation **/
void VLMO_element_addition (VLMO_Operator_Descriptor_t& desc) {

    if (desc.flag_unified_mem == true) {
        VLMO_element_addition_unified (desc);
    }

}

void VLMO_element_addition_unified (VLMO_Operator_Descriptor_t& desc) {

    size_t num_elements = desc.A_h * desc.A_w;
    dim3 threads = desc.num_threads;
    dim3 blocks = desc.num_blocks;

    cuda_element_add<<<blocks, threads>>> (desc.device_A, desc.device_B, desc.device_C, num_elements);
    cudaDeviceSynchronize(); 
    cudaErrChk( cudaGetLastError ());

}


/** Element-wise subtraction operation **/
void VLMO_element_subtraction (VLMO_Operator_Descriptor_t& desc) {
    if (desc.flag_unified_mem == true) {
        VLMO_element_subtraction_unified (desc);
    } 
}

void VLMO_element_subtraction_unified (VLMO_Operator_Descriptor_t& desc) {

    size_t num_elements = desc.A_h * desc.A_w;
    dim3 threads = desc.num_threads;
    dim3 blocks = desc.num_blocks;

    cuda_element_sub<<<blocks, threads>>> (desc.device_A, desc.device_B, desc.device_C, num_elements);
    cudaDeviceSynchronize(); 
    cudaErrChk( cudaGetLastError ());


}


/** Element-wise multiplication operation **/
void VLMO_element_multiplication (VLMO_Operator_Descriptor_t& desc) {
    if (desc.flag_unified_mem == true) {
        VLMO_element_multiplication_unified (desc);
    } 
}

void VLMO_element_multiplication_unified (VLMO_Operator_Descriptor_t& desc) {

    size_t num_elements = desc.A_h * desc.A_w;
    dim3 threads = desc.num_threads;
    dim3 blocks = desc.num_blocks;


    cuda_element_mul<<<blocks, threads>>> (desc.device_A, desc.device_B, desc.device_C, num_elements);
    cudaDeviceSynchronize(); 
    cudaErrChk( cudaGetLastError ());


}

/** Element-wise division operation **/
void VLMO_element_division (VLMO_Operator_Descriptor_t& desc) {
    if (desc.flag_unified_mem == true) {
        VLMO_element_division_unified (desc);
    } 
}

void VLMO_element_division_unified (VLMO_Operator_Descriptor_t& desc) {

    size_t num_elements = desc.A_h * desc.A_w;
    dim3 threads = desc.num_threads;
    dim3 blocks = desc.num_blocks;


    cuda_element_div<<<blocks, threads>>> (desc.device_A, desc.device_B, desc.device_C, num_elements);
    cudaDeviceSynchronize(); 
    cudaErrChk( cudaGetLastError ());


}




/******************************************************
  *****************************************************
  * Functions for matrix multiplication
  *****************************************************
  *******************************************************/

void VLMO_matrix_multiplication (VLMO_Operator_Descriptor_t& desc, VLMO_Operator_t, const bool measure) {


    // Performance measurement
    cudaEvent_t event_start, event_stop;
    if (measure == true) {
        VLMO_record_start (event_start, event_stop);
    }



    if (desc.flag_unified_mem == true) {
        VLMO_matrix_multiplication_unified (desc);
    } 



    // Performance measurement
    if (measure == true) {
        VLMO_record_end (event_start, event_stop);
    }


}

void VLMO_matrix_multiplication_unified (VLMO_Operator_Descriptor_t& desc) {
    
    dim3 threads = desc.num_threads;
    dim3 blocks = desc.num_blocks;

    size_t m = desc.C_h;
    size_t n = desc.B_w;
    size_t k = desc.A_w;

    cuda_matrix_mul_basic<<<blocks, threads>>> (desc.device_A, desc.device_B, desc.device_C, m, n, k);
    cudaDeviceSynchronize(); 
    cudaErrChk( cudaGetLastError ());

}





/******************************************************
  *****************************************************
  * Functions for matrix transposition
  *****************************************************
  *******************************************************/

void VLMO_matrix_transpose (VLMO_Operator_Descriptor_t& desc, VLMO_Operator_t, const bool measure) {
    

    // Performance measurement
    cudaEvent_t event_start, event_stop;
    if (measure == true) {
        VLMO_record_start (event_start, event_stop);
    }



    if (desc.flag_unified_mem == true) {
        VLMO_transposition_unified (desc);
    } 


    

    // Performance measurement
    if (measure == true) {
        VLMO_record_end (event_start, event_stop);
    }

}

void VLMO_matrix_transpose_unified (VLMO_Operator_Descriptor_t& desc) {
    
    dim3 threads = desc.num_threads;
    dim3 blocks = desc.num_blocks;

    size_t m = desc.A_h;
    size_t n = desc.A_w;

    cuda_matrix_transpose_basic<<<blocks, threads>>> (desc.device_A, desc.device_C, m, n);
    cudaDeviceSynchronize(); 
    cudaErrChk( cudaGetLastError ());

}


