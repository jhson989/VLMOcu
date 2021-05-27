
#include "../include/kernel.cuh"

__global__ void cuda_element_add (const float *A, const float *B, float *C, int length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < length) {
        C[i] = A[i] + B[i];
    }
}

__global__ void cuda_element_sub (const float *A, const float *B, float *C, int length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < length) {
        C[i] = A[i] - B[i];
    }
}

__global__ void cuda_element_mul (const float *A, const float *B, float *C, int length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < length) {
        C[i] = A[i] * B[i];
    }
}

__global__ void cuda_element_div (const float *A, const float *B, float *C, int length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < length) {
        if (B[i] != 0)
            C[i] = A[i] / B[i];
        else
            C[i] = 0.0;
    }
}


