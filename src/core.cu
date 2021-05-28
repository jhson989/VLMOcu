#include <cuda.h>
#include "../include/core.cuh"
#include "../include/utils.cuh"




/******************************************************
  * Functions for querying device properties
  *******************************************************/

int VLMO_get_device_num(const bool verbose=false) {
    /*
     * Get the number of GPU device(s) on the machine
     * Args
     *   verbose 
     */
    int num_devices;
    cudaErrChk (cudaGetDeviceCount (&num_devices));

    if (verbose == true) {
        printf("\n=================================================\n");
        printf("The number of device(s) : %d\n", num_devices);
        printf("=================================================\n\n");
    }

    return num_devices;

}



cudaDeviceProp VLMO_get_device_properties(const int device_id, size_t* free, size_t* total, const bool verbose=false) {
    /*
     * Get properties of certain GPU device [device_id]
     * Args
     *   device_id
     *   verbose 
     */

    // get device info
    cudaDeviceProp prop;
    cudaErrChk ( cudaSetDevice (device_id));
    cudaErrChk ( cudaGetDeviceProperties (&prop, device_id) );

    // get memory info
    if (free != NULL && total != NULL) {
        CUdevice dev;
        CUcontext ctx;
        cuDeviceGet(&dev,device_id);
        cuCtxCreate(&ctx, 0, dev);
        cuMemGetInfo (free, total);
    }

    //
    if (verbose == true) {
        printf ("\n========================================================\n");
        printf ("[System Environment]\n");
        printf ("Device Number: %d\n", device_id);
        printf ("  Device name: %s\n", prop.name);
        printf ("  Device compute capability: %d.%d\n", prop.major, prop.minor);
        printf ("  Number of SM(s): %d\n", prop.multiProcessorCount);
        printf ("  Memory Clock Rate (GHz): %.2f\n",
               ((float)prop.memoryClockRate)/1.0e6);
        printf ("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf ("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

        printf ("\n[Kernel size]\n");
        printf ("  Maximum size of a grid [%d, %d, %d]\n"
                , prop.maxGridSize[0], prop.maxGridSize[0], prop.maxGridSize[0]);
        printf ("  Maximum size of a block [%d]\n"
                , prop.maxThreadsPerBlock);

        if (free != NULL && total != NULL) {
            printf ("\n[Global mem]\n");
            printf ("  Global memory size : %.3f GB\n", (float)(*total/1.0e9));
            printf ("  Free memory size : %.3f GB\n", (float)(*free/1.0e9));
        }

        printf ("\n[Shared mem]\n");
        printf ("  Shared memory size per block : %d KB\n", (int)(prop.sharedMemPerBlock/1.0e3));
        printf ("\n========================================================\n");

    }

    return prop;
}





/******************************************************
  * Functions for initiating program 
  *******************************************************/
void VLMO_init (VLMO_Operator_Descriptor_t& desc) {

}



/******************************************************
  * Functions for managing device memory
  *******************************************************/

void VLMO_malloc_device_mem (VLMO_Operator_Descriptor_t& desc, const bool verbose=false) {

//    size_t total_size = sizeof(float)*desc.A_h*desc.A_w + sizeof(float)*desc.B_h*desc.B_w + sizeof(float)*desc.C_h*desc.C_w;

    if (desc.flag_unified_mem == true) {
        VLMO_malloc_device_mem_unified (desc, verbose);
        return ;
    } else {
        VLMO_malloc_device_mem_patch (desc, verbose);
        return ;
    }

}

void VLMO_malloc_device_mem_unified (VLMO_Operator_Descriptor_t& desc, const bool verbose=false) {

    // Allocate unified memory for A
    if (desc.host_A != nullptr) {
        cudaErrChk (cudaMallocManaged (&desc.device_A, sizeof(float)*desc.A_h*desc.A_w));
        memcpy (desc.device_A, desc.host_A, sizeof(float)*desc.A_h*desc.A_w);
        free (desc.host_A);
        desc.host_A = nullptr;
    }

    // Allocate unified memory for B
    if (desc.host_B != nullptr) {
        cudaErrChk (cudaMallocManaged (&desc.device_B, sizeof(float)*desc.B_h*desc.B_w));
        memcpy (desc.device_B, desc.host_B, sizeof(float)*desc.B_h*desc.B_w);
        free (desc.host_B);
        desc.host_B = nullptr;
    }

    // Allocate unified memory for C
    if (desc.host_C != nullptr) {
        cudaErrChk (cudaMallocManaged (&desc.device_C, sizeof(float)*desc.C_h*desc.C_w));
        memcpy (desc.device_C, desc.host_C, sizeof(float)*desc.C_h*desc.C_w);
        free (desc.host_C);
        desc.host_C = nullptr;
    }

    if (verbose == true) {
        size_t total_size = sizeof(float)*desc.A_h*desc.A_w + sizeof(float)*desc.B_h*desc.B_w + sizeof(float)*desc.C_h*desc.C_w;
        printf("[Mem] Unified memory allocation completed..\n");
        printf("    mem usage : %.3f GB [free : %.3f GB]\n", total_size*1e-9, desc.mem_free_size*1e-9);
    }
}

void VLMO_malloc_device_mem_patch (VLMO_Operator_Descriptor_t& desc, const bool verbose=false) {
        
    desc.flag_double_buffering = true;
    cudaErrChk (cudaStreamCreate (&desc.streams[0]));
    cudaErrChk (cudaStreamCreate (&desc.streams[1]));

    if (desc.patch_h == 0 || desc.patch_w == 0) {
        get_maximum_size_patch (desc);
    }

    size_t total_size_patch = sizeof (float) * desc.patch_h * desc.patch_w * 2;

    // Allocate unified memory for A
    if (desc.host_A != nullptr) {
        cudaErrChk (cudaMalloc (&desc.device_A, total_size_patch));
        cudaErrChk (cudaMemcpyAsync (desc.device_A, desc.host_A, total_size_patch/2, cudaMemcpyHostToDevice, desc.streams[0]));
    }

    // Allocate unified memory for B
    if (desc.host_B != nullptr) {
        cudaErrChk (cudaMalloc (&desc.device_B, total_size_patch));
        cudaErrChk (cudaMemcpyAsync (desc.device_B, desc.host_B, total_size_patch/2, cudaMemcpyHostToDevice, desc.streams[0]));
    }

    // Allocate unified memory for C
    if (desc.host_C != nullptr) {
        cudaErrChk (cudaMalloc (&desc.device_C, total_size_patch));
    }

    cudaErrChk (cudaStreamSynchronize (desc.streams[0]));
    cudaErrChk (cudaGetLastError ());

    if (verbose == true) {
        printf("[Mem] Patch memory allocation completed..\n");
        printf("    mem usage : %.3f GB [free : %.3f GB]\n", total_size_patch*1e-9, desc.mem_free_size*1e-9);
    }


}





void VLMO_clear_all (VLMO_Operator_Descriptor_t& desc) {

    if (desc.host_A != nullptr)
        free (desc.host_A);

    if (desc.device_A != nullptr)  
        cudaErrChk (cudaFree (desc.device_A));

    if (desc.host_B != nullptr)
        free (desc.host_B);

    if (desc.device_B != nullptr)  
        cudaErrChk (cudaFree (desc.device_B));

    if (desc.host_C != nullptr)
        free (desc.host_C); 

    if (desc.device_C != nullptr)  
        cudaErrChk (cudaFree (desc.device_C));

    if (desc.flag_double_buffering == true) {
        cudaStreamDestroy(desc.streams[0]);
        cudaStreamDestroy(desc.streams[1]);
    }

}

