#include <cuda.h>
#include "../include/core.cuh"




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

    }

    return prop;
}




/******************************************************
  * Functions for managing device memory
  *******************************************************/

void VLMO_malloc_device_mem (VLMO_Operator_Descriptor_t& desc) {

    if (desc.flag_unified_mem == true) {
        VLMO_malloc_device_mem_unified (desc);
        return ;
    } 

}

void VLMO_malloc_device_mem_unified (VLMO_Operator_Descriptor_t& desc) {

    // Allocate unified memory for A
    cudaErrChk( cudaMallocManaged (&desc.device_A, sizeof(float)*desc.A_h*desc.A_w));
    memcpy (desc.device_A, desc.host_A, sizeof(float)*desc.A_h*desc.A_w);

    // Allocate unified memory for B
    cudaErrChk( cudaMallocManaged (&desc.device_B, sizeof(float)*desc.B_h*desc.B_w));
    memcpy (desc.device_B, desc.host_B, sizeof(float)*desc.B_h*desc.B_w);

    // Allocate unified memory for C
    cudaErrChk( cudaMallocManaged (&desc.device_C, sizeof(float)*desc.C_h*desc.C_w));
    memcpy (desc.device_C, desc.host_C, sizeof(float)*desc.C_h*desc.C_w);


    printf("1 End..\n");
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
    
    printf("2 End..\n");
}

