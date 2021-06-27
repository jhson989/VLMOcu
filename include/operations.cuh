#ifndef __OPERATIONS__
#define __OPERATIONS__

#include "core.cuh"

/** Matrix element-wise operations **/
void VLMO_element_operation (VLMO_Operator_Descriptor_t& desc, const bool measure);
void VLMO_element_unified (VLMO_Operator_Descriptor_t& desc);
void VLMO_element_patch (VLMO_Operator_Descriptor_t& desc);


/** Matrix multiplication **/
void VLMO_matrix_multiplication (VLMO_Operator_Descriptor_t& desc, const bool measure);
void VLMO_matrix_multiplication_unified (VLMO_Operator_Descriptor_t& desc);
void VLMO_matrix_multiplication_patch (VLMO_Operator_Descriptor_t& desc);
void VLMO_memcpy_patch(VLMO_Operator_Descriptor_t& desc, float* A, float* B, int mode, int idx_mem, const size_t H_0, const size_t W_0, const size_t len, const size_t max_h, const size_t max_w);
void _VLMO_matrix_mul_patch (VLMO_Operator_Descriptor_t& desc, dim3 blocks, dim3 threads, cudaStream_t& stream, const float *A, const float *B, float *C, const int M, const int N, const int K, const int patch_h, const int patch_w, const int patch_k, const int patch_start_h, const int patch_start_w, const int patch_start_k) ;
void VLMO_matrix_transpose (VLMO_Operator_Descriptor_t& desc, const bool measure);
void VLMO_matrix_transpose_unified (VLMO_Operator_Descriptor_t& desc);






#endif
