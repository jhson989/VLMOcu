
#include "../include/VLMO_type.cuh"
#include "../include/VLMO_struct.cuh"

int VLMO_get_device_num(const bool);

/***
  ** Functions for querying device properties
  ***/

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

