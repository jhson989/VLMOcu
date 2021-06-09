
#include <cstdio>
#include "../include/kernel.cuh"

__global__ void cuda_element_add (const float *A, const float *B, float *C, int length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < length) {
        C[i] = A[i] + B[i];
    }
}


__global__ void cuda_element_add_patch (const float *A, const float *B, float *C,  const size_t done, const size_t num_elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (done+i < num_elements) {
        C[i] = A[i] + B[i];
    }
}



__global__ void cuda_element_sub (const float *A, const float *B, float *C, int length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < length) {
        C[i] = A[i] - B[i];
    }
}

__global__ void cuda_element_sub_patch (const float *A, const float *B, float *C,  const size_t done, const size_t num_elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (done+i < num_elements) {
        C[i] = A[i] - B[i];
    }
}


__global__ void cuda_element_mul (const float *A, const float *B, float *C, int length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < length) {
        C[i] = A[i] * B[i];
    }
}

__global__ void cuda_element_mul_patch (const float *A, const float *B, float *C,  const size_t done, const size_t num_elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (done+i < num_elements) {
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

__global__ void cuda_element_div_patch (const float *A, const float *B, float *C,  const size_t done, const size_t num_elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (done+i < num_elements) {
        if (B[i] != 0)
            C[i] = A[i] / B[i];
        else
            C[i] = 0.0;
    }
}




/******************************************************
  *****************************************************
  * CUDA kernels for matrix multiplication
  *****************************************************
  *******************************************************/

__global__ void cuda_matrix_mul_basic (const float *A, const float *B, float *C, const size_t M, const size_t N, const size_t K) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i<M&&j<N) {
        float sum=0;
        for (int l=0; l<K; l++) {
            sum += A[i*K+l]*B[l*K+j];
        }
        C[i*K+j]=sum;
    }
}

__global__ void cuda_matrix_mul_patch (const float *A, const float *B, float *C, const size_t M, const size_t N, const size_t K, const size_t patch_h, const size_t patch_w, const size_t patch_k, const size_t patch_start_h, const size_t patch_start_w, const size_t patch_start_k) {



    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i<patch_h) && (i+patch_start_h<M) && (j<patch_w) && (j+patch_start_w<N)) {
        
        float sum=0;
        for (int l=0; (l<patch_k) && (patch_start_k+l<K); l++) {
            sum += A[i*patch_k+l]*B[l*patch_k+j];
        }
        C[i*patch_w+j]+=sum;
    }

}

__global__ void cuda_matrix_mul_patch_tiled (const float *A, const float *B, float *C, const size_t M, const size_t N, const size_t K, const size_t patch_h, const size_t patch_w, const size_t patch_k, const size_t patch_start_h, const size_t patch_start_w, const size_t patch_start_k, const size_t len_tile) {


    extern __shared__ float smem[];
    float *sA = &smem[0];
    float *sB = &smem[len_tile*len_tile];

    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int si = threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sj = threadIdx.x;
    
    
    
    float sum=0;
    for (int l=0; (l<patch_k) && (patch_start_k+l<K); l+=len_tile) {
        //sum += A[i*patch_k+l]*B[l*patch_k+j];

        if ((patch_start_k+(l+sj)<K))
            sA[si*len_tile+sj] = A[i*patch_k+(l+sj)];
        else
            sA[si*len_tile+sj] = 0;
        if ((patch_start_k+(l+si)<K))
            sB[si*len_tile+sj] = B[(l+si)*patch_k+j];
        else
            sB[si*len_tile+sj] = 0;

        __syncthreads();

        for (int k=0; k<len_tile; k++) {
            sum += sA[(si*len_tile)+k]*sB[(k*len_tile)+sj];
        }
        __syncthreads();
    }
    if ((i<patch_h) && (i+patch_start_h<M) && (j<patch_w) && (j+patch_start_w<N)) {
        C[i*patch_w+j]+=sum;
    }

}


/******************************************************
  *****************************************************
  * CUDA kernels for matrix transposition
  *****************************************************
  *******************************************************/
  


__global__ void cuda_matrix_transpose_basic (const float *in, float *out, const size_t M, const size_t N) {
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    out[j*M+i] = in[i*N+j];
}
