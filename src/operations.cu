
#include "../include/operations.cuh"
#include "../include/kernel.cuh"
#include "../include/utils.cuh"








/******************************************************
  *****************************************************
  * Functions for element-wise opearations
  *****************************************************
  *******************************************************/

void VLMO_element_operation (VLMO_Operator_Descriptor_t& desc, VLMO_Operator_t op=VLMO_Op_No, const bool measure=false) {

    // Performance measurement
    cudaEvent_t event_start, event_stop;
    if (measure == true) {
        VLMO_record_start (event_start, event_stop);
    }


    if (desc.flag_unified_mem == true) {
        VLMO_element_unified (desc);
    } else {
        VLMO_element_patch (desc);
    }
    
    
    // Performance measurement
    if (measure == true) {
        VLMO_record_end (event_start, event_stop);
    }

}



/*******************************************************************/
/** Element-wise addtion operation **/
void VLMO_element_unified (VLMO_Operator_Descriptor_t& desc) {

    size_t num_elements = desc.A_h * desc.A_w;
    dim3 threads = desc.num_threads;
    dim3 blocks = dim3((desc.A_w*desc.A_h+desc.num_threads.x-1) / desc.num_threads.x);

    switch (desc.op) {
        case VLMO_Op_Element_Add:
            cuda_element_add<<<blocks, threads>>> (desc.device_A[0], desc.device_B[0], desc.device_C[0], num_elements);
            break;
        case VLMO_Op_Element_Sub:
            cuda_element_sub<<<blocks, threads>>> (desc.device_A[0], desc.device_B[0], desc.device_C[0], num_elements);
            break;
        case VLMO_Op_Element_Mul:
            cuda_element_mul<<<blocks, threads>>> (desc.device_A[0], desc.device_B[0], desc.device_C[0], num_elements);
            break;
        case VLMO_Op_Element_Div:
            cuda_element_div<<<blocks, threads>>> (desc.device_A[0], desc.device_B[0], desc.device_C[0], num_elements);
            break;
    }


    cudaDeviceSynchronize(); 
    cudaErrChk (cudaGetLastError ());

}

void VLMO_element_patch (VLMO_Operator_Descriptor_t& desc) {

    size_t num_total_elements = desc.A_h * desc.A_w;
    size_t num_patch_elements = desc.patch_h * desc.patch_w;
    size_t num_process=0, num_process_pre=0, num_process_next=0;
    size_t done=0, done_pre=0, done_next=0;
    dim3 threads = desc.num_threads;
    dim3 blocks = dim3((num_patch_elements+desc.num_threads.x-1) / desc.num_threads.x);
    bool idx_mem=true;


    /**********************************************************/
    /** Transfer first data from host to device **/
    done_next = done + num_process;
    num_process_next = num_patch_elements;
    if (done_next+num_process_next >= num_total_elements) num_process_next = num_total_elements - done_next;
    if (num_process_next > 0) {
        printf("   HtoD: H[%lu, %lu](%d) to (%lu, %lu)\n", done_next/desc.A_w, done_next%desc.A_w, (int)!idx_mem, (done_next+num_process_next)/desc.A_w, (done_next+num_process_next)%desc.A_w);
        cudaErrChk (cudaMemcpyAsync (desc.device_A[(int)!idx_mem], &(desc.host_A[done_next]), num_process_next*sizeof (float), cudaMemcpyHostToDevice, desc.streams[0]));
        cudaErrChk (cudaMemcpyAsync (desc.device_B[(int)!idx_mem], &(desc.host_B[done_next]), num_process_next*sizeof (float), cudaMemcpyHostToDevice, desc.streams[0]));
    }

    for (done=0; done<num_total_elements; done+=num_process) {

        /**********************************************************/
        /** Calculate pre, now, next; **/
        idx_mem = !idx_mem;
        // Previous
        done_pre = done-num_process;
        num_process_pre = num_process;
        // Now
        num_process = num_patch_elements;
        if (done+num_process >= num_total_elements) num_process = num_total_elements - done;
        // Next
        done_next = done + num_process;
        num_process_next = num_patch_elements;
        if (done_next+num_process_next >= num_total_elements) num_process_next = num_total_elements - done_next;


        /**********************************************************/
        /** Launch kernel **/
        switch (desc.op) {
            case VLMO_Op_Element_Add:
                cuda_element_add_patch<<<blocks, threads, 0, desc.streams[1]>>> (desc.device_A[(int)idx_mem], desc.device_B[(int)idx_mem], desc.device_C[(int)idx_mem], done, num_total_elements);
                break;
            case VLMO_Op_Element_Sub:
                cuda_element_sub_patch<<<blocks, threads, 0, desc.streams[1]>>> (desc.device_A[(int)idx_mem], desc.device_B[(int)idx_mem], desc.device_C[(int)idx_mem], done, num_total_elements);
                break;
            case VLMO_Op_Element_Mul:
                cuda_element_mul_patch<<<blocks, threads, 0, desc.streams[1]>>> (desc.device_A[(int)idx_mem], desc.device_B[(int)idx_mem], desc.device_C[(int)idx_mem], done, num_total_elements);
                break;
            case VLMO_Op_Element_Div:
                cuda_element_div_patch<<<blocks, threads, 0, desc.streams[1]>>> (desc.device_A[(int)idx_mem], desc.device_B[(int)idx_mem], desc.device_C[(int)idx_mem], done, num_total_elements);
                break;
        }

        /**********************************************************/
        /** Transfer input(reseult) data from host(device) to device(host) **/
        if (num_process_pre > 0) {
            cudaErrChk (cudaMemcpyAsync (&(desc.host_C[done_pre]), desc.device_C[(int)!idx_mem], num_process_pre*sizeof (float), cudaMemcpyDeviceToHost, desc.streams[0]));
        }
        if (num_process_next > 0) {
            cudaErrChk (cudaMemcpyAsync (desc.device_A[(int)!idx_mem], &(desc.host_A[done_next]), num_process_next*sizeof (float), cudaMemcpyHostToDevice, desc.streams[0]));
            cudaErrChk (cudaMemcpyAsync (desc.device_B[(int)!idx_mem], &(desc.host_B[done_next]), num_process_next*sizeof (float), cudaMemcpyHostToDevice, desc.streams[0]));
        }
        cudaErrChk (cudaStreamSynchronize (desc.streams[0]));
        cudaErrChk (cudaStreamSynchronize (desc.streams[1]));
        cudaErrChk (cudaGetLastError ());
    }


    /**********************************************************/
    /** Transfer last result from device to host **/
    done_pre = done-num_process;
    num_process_pre = num_process;
    if (num_process_pre > 0) {
        cudaErrChk (cudaMemcpyAsync (&desc.host_C[done_pre], desc.device_C[(int)(idx_mem)], num_process_pre*sizeof (float), cudaMemcpyDeviceToHost, desc.streams[0]));
    }
    cudaErrChk (cudaStreamSynchronize (desc.streams[0]));
    cudaErrChk (cudaGetLastError ());
}




/******************************************************
  *****************************************************
  * Functions for matrix multiplication
  *****************************************************
  *******************************************************/

void VLMO_matrix_multiplication (VLMO_Operator_Descriptor_t& desc, VLMO_Operator_t, const bool measure) {


    // Performance measurement
    cudaEvent_t event_start, event_stop;
    if (measure == true) {
        VLMO_record_start (event_start, event_stop);
    }



    if (desc.flag_unified_mem == true) {
        VLMO_matrix_multiplication_unified (desc);
    } 



    // Performance measurement
    if (measure == true) {
        VLMO_record_end (event_start, event_stop);
    }


}

void VLMO_matrix_multiplication_unified (VLMO_Operator_Descriptor_t& desc) {
    
    dim3 threads = desc.num_threads;
    dim3 blocks = dim3((desc.C_h+desc.num_threads.x-1) / desc.num_threads.x, (desc.C_w+desc.num_threads.y-1) / desc.num_threads.y);

    size_t m = desc.C_h;
    size_t n = desc.B_w;
    size_t k = desc.A_w;

    cuda_matrix_mul_basic<<<blocks, threads>>> (desc.device_A[0], desc.device_B[0], desc.device_C[0], m, n, k);
    cudaDeviceSynchronize(); 
    cudaErrChk( cudaGetLastError ());

}





/******************************************************
  *****************************************************
  * Functions for matrix transposition
  *****************************************************
  *******************************************************/

void VLMO_matrix_transpose (VLMO_Operator_Descriptor_t& desc, VLMO_Operator_t, const bool measure) {
    

    // Performance measurement
    cudaEvent_t event_start, event_stop;
    if (measure == true) {
        VLMO_record_start (event_start, event_stop);
    }



    if (desc.flag_unified_mem == true) {
        VLMO_matrix_transpose_unified (desc);
    } 


    

    // Performance measurement
    if (measure == true) {
        VLMO_record_end (event_start, event_stop);
    }

}

void VLMO_matrix_transpose_unified (VLMO_Operator_Descriptor_t& desc) {
    
    dim3 threads = desc.num_threads;
    dim3 blocks = dim3((desc.C_h+desc.num_threads.x-1) / desc.num_threads.x, (desc.C_w+desc.num_threads.y-1) / desc.num_threads.y);

    size_t m = desc.A_h;
    size_t n = desc.A_w;

    cuda_matrix_transpose_basic<<<blocks, threads>>> (desc.device_A[0], desc.device_C[0], m, n);
    cudaDeviceSynchronize(); 
    cudaErrChk( cudaGetLastError ());

}


