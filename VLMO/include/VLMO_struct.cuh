
#ifndef __VLMO_STRUCT__
#define __VLMO_STRUCT__

#include "VLMO_type.cuh"

/***
  ** Data structure for describing a target matrix operation
  ***/
template <typename T>
typedef struct {

    /** Matrix Operation **/
    // (C = A op B) or (C= op A)
    vlmoOperator_t op = VLMO_OP_NO;

    /** Matrices Declaration **/
    // input A
    T* host_A = nullptr;
    T* device_A[2] = {nullptr, nullptr};
    size_t AW=0, AH=0;

    // input B
    T* host_B = nullptr;
    T* device_B[2] = {nullptr, nullptr};
    size_t BW=0, BH=0;

    // output C
    T* host_C = nullptr;
    T* device_C[2] = {nullptr, nullptr};
    size_t CW=0, CH=0;

    /** Device Properties **/
    size_t num_device=1;
    cudaDeviceProp prop;
    cudaStream_t streams[2];

    /** Memory manage **/
    vlmoMem_t mode_malloc=VLMO_MEM_NO;
    size_t patchW=0, patchH=0;

a

} vlmoOperatorDescriptor_t;


#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#endif
