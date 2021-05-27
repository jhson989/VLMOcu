
#ifndef __KERNEL__
#define __KERNEL__

__global__ void cuda_element_add (const float *A, const float *B, float *C, int length);
__global__ void cuda_element_sub (const float *A, const float *B, float *C, int length);
__global__ void cuda_element_mul (const float *A, const float *B, float *C, int length);
__global__ void cuda_element_div (const float *A, const float *B, float *C, int length);


#endif
