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
