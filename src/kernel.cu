
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

__global__ void cuda_matrix_mul_patch (const float *A, const float *B, float *C, const int M, const int N, const int K, const int A_w, const int B_w) {



    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<M && j<N) {
        float sum=0;
        for (int l=0; l<K; l++) {
            sum += A[i*A_w+l]*B[l*B_w+j];
        }
        C[i*B_w+j]+=sum;
    }

}

__global__ void cuda_matrix_mul_patch_tiled (const float *A, const float *B, float *C, const int M, const int N, const int K, const int A_w, const int B_w) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int len_tile = blockDim.x, si=threadIdx.y, sj=threadIdx.x;
    int sidx = si*len_tile+sj;

    extern __shared__ float smem[];
    float *sA = &smem[0];
    float *sB = &smem[len_tile*len_tile];
    
    float sum = 0.f;
    for (int tile=0; tile<K; tile+=len_tile) {
        
        if (tile+sj<K && i<M)
            sA[sidx] = A[i*A_w+(tile+sj)];
        else
            sA[sidx] = 0.f;
        if (tile+si<K && j<N)
            sB[sidx] = B[(tile+si)*B_w+j];
        else 
            sB[sidx] = 0.f;
        __syncthreads();
        for (int k=0; k<len_tile; k++)
            sum += sA[si*len_tile+k]*sB[k*len_tile+sj];
        __syncthreads();
    }
    if (i<M && j<N)
        C[i*B_w+j] += sum;
}

__global__ void cuda_matrix_mul_patch_tiled_full_loaded (const float *A, const float *B, float *C, const int M, const int N, const int K, const int A_w, const int B_w) {

    int len_tile = blockDim.x, si=threadIdx.y, sj=threadIdx.x;
    int sidx = si*len_tile+sj;
    extern __shared__ float smem[];
    float *sA = &smem[0];
    float *sB = &smem[len_tile*len_tile];
    
    int i, j;
    float sum;


    i = blockIdx.y * blockDim.y + threadIdx.y;
    j = blockIdx.x * blockDim.x + threadIdx.x;
    sum = 0.f;
    for (int tile=0; tile<K; tile+=len_tile) {
        sA[sidx] = A[i*A_w+(tile+sj)];
        sB[sidx] = B[(tile+si)*B_w+j];
        __syncthreads();
        for (int k=0; k<len_tile; k++)
            sum += sA[si*len_tile+k]*sB[k*len_tile+sj];
        __syncthreads();
    }
    C[i*B_w+j] += sum;
    __syncthreads();


    i = blockIdx.y * blockDim.y + threadIdx.y;
    j = blockIdx.x * blockDim.x + (threadIdx.x + gridDim.x*blockDim.x);
    sum = 0.f;
    for (int tile=0; tile<K; tile+=len_tile) {
        sA[sidx] = A[i*A_w+(tile+sj)];
        sB[sidx] = B[(tile+si)*B_w+j];
        __syncthreads();
        for (int k=0; k<len_tile; k++)
            sum += sA[si*len_tile+k]*sB[k*len_tile+sj];
        __syncthreads();
    }
    C[i*B_w+j] += sum;
    __syncthreads();


    
    i = (blockIdx.y + gridDim.y) * blockDim.y + threadIdx.y;
    j = blockIdx.x * blockDim.x + threadIdx.x;
    sum = 0.f;
    for (int tile=0; tile<K; tile+=len_tile) {
        sA[sidx] = A[i*A_w+(tile+sj)];
        sB[sidx] = B[(tile+si)*B_w+j];
        __syncthreads();
        for (int k=0; k<len_tile; k++)
            sum += sA[si*len_tile+k]*sB[k*len_tile+sj];
        __syncthreads();
    }
    C[i*B_w+j] += sum;
    __syncthreads();


    i = (blockIdx.y + gridDim.y) * blockDim.y + threadIdx.y;
    j = blockIdx.x * blockDim.x + (threadIdx.x + gridDim.x*blockDim.x);
    sum = 0.f;
    for (int tile=0; tile<K; tile+=len_tile) {
        sA[sidx] = A[i*A_w+(tile+sj)];
        sB[sidx] = B[(tile+si)*B_w+j];
        __syncthreads();
        for (int k=0; k<len_tile; k++)
            sum += sA[si*len_tile+k]*sB[k*len_tile+sj];
        __syncthreads();
    }
    C[i*B_w+j] += sum;
    


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
