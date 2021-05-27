
#include "../include/operations.cuh"
#include "../include/kernel.cuh"
#include "../include/utils.cuh"








/******************************************************
  * Functions for element-wise opearations
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
    int num_threads = 1024;
    int num_blocks = (num_elements+num_threads-1) / num_threads;

    cuda_element_add<<<num_blocks, num_threads>>> (desc.device_A, desc.device_B, desc.device_C, num_elements);
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
    int num_threads = 1024;
    int num_blocks = (num_elements+num_threads-1) / num_threads;

    cuda_element_sub<<<num_blocks, num_threads>>> (desc.device_A, desc.device_B, desc.device_C, num_elements);
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
    int num_threads = 1024;
    int num_blocks = (num_elements+num_threads-1) / num_threads;

    cuda_element_mul<<<num_blocks, num_threads>>> (desc.device_A, desc.device_B, desc.device_C, num_elements);
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
    int num_threads = 1024;
    int num_blocks = (num_elements+num_threads-1) / num_threads;

    cuda_element_div<<<num_blocks, num_threads>>> (desc.device_A, desc.device_B, desc.device_C, num_elements);
    cudaDeviceSynchronize(); 
    cudaErrChk( cudaGetLastError ());


}




/******************************************************
  * Functions for matrix multiplication
  *******************************************************/

void VLMO_matrix_multiplication (VLMO_Operator_Descriptor_t& desc) {
    if (desc.flag_unified_mem == true) {
        VLMO_matrix_multiplication_unified (desc);
    } 
}

void VLMO_matrix_multiplication_unified (VLMO_Operator_Descriptor_t& desc) {

}





/******************************************************
  * Functions for matrix transposition
  *******************************************************/

void VLMO_transposition (VLMO_Operator_Descriptor_t& desc) {
    if (desc.flag_unified_mem == true) {
        VLMO_transposition_unified (desc);
    } 
}

void VLMO_transposition_unified (VLMO_Operator_Descriptor_t& desc) {

}


