
#ifndef __CORE__
#define __CORE__

#include <cstdlib>
#include <cstdio>

// Function declaration

typedef enum {
   VLMO_Add = 0, 
   VLMO_Subtract,
   VLMO_Multiply,
   VLMO_Transpose
} VLMO_Operator_t;

typedef enum {
   double_buffering = 0
} Optim_t;


typedef struct {

   // C = A op B 
   VLMO_Operator_t op;
   
   // A
   size_t A_h=0;
   size_t A_w=0;
   float* A=NULL;

   // B
   size_t B_h=0;
   size_t B_w=0;
   float* B=NULL;

   // C
   size_t C_h=0;
   size_t C_w=0;
   float* C=NULL;

   // Optim
   bool flag_unified_mem=false;

   // Device properties
   size_t num_device=0;
   size_t mem_free_size=0;


} VLMO_Operator_Descriptor_t;



// Function declaration

int VLMO_get_device_num(const bool verbose);
cudaDeviceProp VLMO_get_device_properties(const int device_id, size_t* free, size_t* total, const bool verbose);

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


