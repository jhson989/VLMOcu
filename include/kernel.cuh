
#ifndef __KERNEL__
#define __KERNEL__

/** Element-wise operations **/
__global__ void cuda_element_add (const float *A, const float *B, float *C, int length);
__global__ void cuda_element_add_patch (const float *A, const float *B, float *C, const size_t done, const size_t num_elements);
__global__ void cuda_element_sub (const float *A, const float *B, float *C, int length);
__global__ void cuda_element_mul (const float *A, const float *B, float *C, int length);
__global__ void cuda_element_div (const float *A, const float *B, float *C, int length);

/** Matrix multiplication **/
__global__ void cuda_matrix_mul_basic (const float *A, const float *B, float *C, const size_t M, const size_t N, const size_t K);

/** Matrix transposition **/
__global__ void cuda_matrix_transpose_basic (const float *in, float *out, const size_t M, const size_t N);

#endif
