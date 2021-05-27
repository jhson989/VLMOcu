
#ifndef __CORE__
#define __CORE__

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>

// Function declaration

typedef enum {

    // Element-wise operation
    VLMO_Op_Element_Add= 0, 
    VLMO_Op_Element_Sub,
    VLMO_Op_Element_Mul,
    VLMO_Op_Element_Div,

    // Matrix multiplication
    VLMO_Op_Mat_Mul,

    // Single matrix operation
    VLMO_Op_Transpose,

    // Dummy operation
    VLMO_Op_No

} VLMO_Operator_t;

const std::string VLMO_Op_Name[] = {
    "VLMO_Op_Element_Add", // 0
    "VLMO_Op_Element_Sub",
    "VLMO_Op_Element_Mul",
    "VLMO_Op_Element_Div",
    "VLMO_Op_Mat_MuL",
    "VLMO_Op_Transpose",
    "VLMO_Op_No"
};

typedef enum {
   double_buffering = 0
} Optim_t;


typedef struct {

   // C = A op B 
   VLMO_Operator_t op;
   
   // A
   size_t A_h=0;
   size_t A_w=0;
   float* host_A=nullptr;
   float* device_A=nullptr;

   // B
   size_t B_h=0;
   size_t B_w=0;
   float* host_B=nullptr;
   float* device_B=nullptr;

   // C
   size_t C_h=0;
   size_t C_w=0;
   float* host_C=nullptr;
   float* device_C=nullptr;

   // Optim
   dim3 num_threads;
   dim3 num_blocks;
   bool flag_unified_mem=false;
   bool flag_double_buffering=false;
   size_t patch_h=-1;
   size_t patch_w=-1;

   // Device properties
   size_t num_device=0;
   size_t mem_free_size=0;
   cudaDeviceProp prop;
   cudaStream_t streams[2];


} VLMO_Operator_Descriptor_t;



// Function declaration
int VLMO_get_device_num(const bool verbose);
cudaDeviceProp VLMO_get_device_properties(const int device_id, size_t* free, size_t* total, const bool verbose);
void VLMO_init (VLMO_Operator_Descriptor_t& desc);
void VLMO_malloc_device_mem (VLMO_Operator_Descriptor_t& desc, const bool verbose);
void VLMO_malloc_device_mem_unified (VLMO_Operator_Descriptor_t& desc, const bool verbose);
void VLMO_malloc_device_mem_patch (VLMO_Operator_Descriptor_t& desc, const bool verbose);
void VLMO_clear_all (VLMO_Operator_Descriptor_t& desc);

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


