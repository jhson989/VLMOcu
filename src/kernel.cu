


__global__ void cuda_matrix_add (const float *A, const float *B, float *C, int length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < length) {
        C[i] = A[i] + B[i];
    }
}