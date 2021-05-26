
#include "../include/operations.cuh"
#include "../include/kernel.cuh"
#include "../include/utils.cuh"



/******************************************************
  * Functions for matrix addition
  *******************************************************/

void VLMO_addition (VLMO_Operator_Descriptor_t& desc, const bool measure=false) {


    // Performance measurement
    cudaEvent_t event_start, event_stop;
    if (measure == true) {
        VLMO_record_start (event_start, event_stop);
    }


    // Main Body
    // Do a operation according to flags
    if (desc.flag_unified_mem == true) {
        VLMO_addition_unified (desc);
    }

    // Performance measurement
    if (measure == true) {
        VLMO_record_end (event_start, event_stop);
    }


}

void VLMO_addition_unified (VLMO_Operator_Descriptor_t& desc){

    size_t num_elements = desc.A_h * desc.A_w;
    int num_threads = 1024;
    int num_blocks = (num_elements+num_threads-1) / num_threads;

    cuda_matrix_add<<<num_blocks, num_threads>>> (desc.device_A, desc.device_B, desc.device_C, num_elements);
    cudaDeviceSynchronize(); 
    cudaErrChk( cudaGetLastError ());

}





/******************************************************
  * Functions for matrix subtraction
  *******************************************************/

void VLMO_subtraction (VLMO_Operator_Descriptor_t& desc, const bool measure=false) {
    if (desc.flag_unified_mem == true) {
        VLMO_subtraction_unified (desc);
    } 
}

void VLMO_subtraction_unified (VLMO_Operator_Descriptor_t& desc) {

}




/******************************************************
  * Functions for matrix multiplication
  *******************************************************/

void VLMO_multiplication (VLMO_Operator_Descriptor_t& desc, const bool measure=false) {
    if (desc.flag_unified_mem == true) {
        VLMO_multiplication_unified (desc);
    } 
}

void VLMO_multiplication_unified (VLMO_Operator_Descriptor_t& desc) {


}





/******************************************************
  * Functions for matrix transposition
  *******************************************************/

void VLMO_transposition (VLMO_Operator_Descriptor_t& desc, const bool measure=false) {
    if (desc.flag_unified_mem == true) {
        VLMO_transposition_unified (desc);
    } 
}

void VLMO_transposition_unified (VLMO_Operator_Descriptor_t& desc) {

}


