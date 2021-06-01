
#include "../include/operations.cuh"
#include "../include/kernel.cuh"
#include "../include/utils.cuh"








/******************************************************
  *****************************************************
  * Functions for element-wise opearations
  *****************************************************
  *******************************************************/

void VLMO_element_operation (VLMO_Operator_Descriptor_t& desc, const bool measure=false) {

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

void VLMO_matrix_multiplication (VLMO_Operator_Descriptor_t& desc, const bool measure) {


    // Performance measurement
    cudaEvent_t event_start, event_stop;
    if (measure == true) {
        VLMO_record_start (event_start, event_stop);
    }



    if (desc.flag_unified_mem == true) {
        VLMO_matrix_multiplication_unified (desc);
    } else {
        VLMO_matrix_multiplication_patch (desc);
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


void VLMO_matrix_multiplication_patch (VLMO_Operator_Descriptor_t& desc) {
    
    dim3 threads = desc.num_threads;
    dim3 blocks = dim3((desc.C_h+desc.num_threads.x-1) / desc.num_threads.x, (desc.C_w+desc.num_threads.y-1) / desc.num_threads.y);

    bool idx_mem_C=true, idx_mem_AB=true;
    size_t patch_h=desc.patch_h, patch_w=desc.patch_w;
    size_t patch_start_h=0, patch_start_w=0, patch_start_k;
    size_t patch_start_h_pre=0, patch_start_w_pre=0;
    size_t patch_start_h_next=0, patch_start_w_next=0;
    

    size_t m = desc.C_h;
    size_t n = desc.B_w;
    size_t k = desc.A_w;


    // Send first input data from host to device
    
    for (patch_start_h=0; patch_start_h<desc.C_h; patch_start_h+=patch_h) {
        for (patch_start_w=0; patch_start_w<desc.C_w; patch_start_w+=patch_w) {         

            /** Update next state **/
            if (patch_start_w + patch_w >= desc.C_w) patch_start_w_next = 0; else patch_start_w_next = patch_start_w + patch_w;
            if (patch_start_w + patch_w >= desc.C_w) patch_start_h_next = patch_start_h + patch_h; else patch_start_h_next = patch_start_h;
            patch_start_k = 0;

            idx_mem_C = !idx_mem_C;
            cudaErrChk (cudaMemset (desc.device_C[idx_mem_C], 0, patch_w*patch_h));
            VLMO_memcpy_patch(desc.device_A[idx_mem_AB], desc.host_A, patch_start_h, patch_h, patch_start_k, patch_w, VLMO_Memcpy_HtoD, idx_mem_AB);
            VLMO_memcpy_patch(desc.device_B[idx_mem_AB], desc.host_B, patch_start_k, patch_h, patch_start_w, patch_w, VLMO_Memcpy_HtoD, idx_mem_AB);

            for (patch_start_k=0; patch_start_k<k; patch_start_k+=patch_w) {

                printf("\n");
                /** Synchronize device **/
                cudaErrChk (cudaStreamSynchronize (desc.streams[0]));
                cudaErrChk (cudaStreamSynchronize (desc.streams[1]));
                cudaErrChk (cudaGetLastError ());

                /** Launch kernel : stream #1 **/
                printf("do [h%lu w%lu k%lu][AB%d][C%d]\n", patch_start_h, patch_start_w, patch_start_k, (int)idx_mem_AB, (int)idx_mem_C);
                //cuda_matrix_mul_patch<<<blocks, threads 0, desc.streams[1]>>> (desc.device_A[(int)idx_mem], desc.device_B[(int)idx_mem], desc.device_C[(int)idx_mem], m, n, k, patch_h, patch_w, patch_start_k);    
                
                /** Send data from host to device **/
                idx_mem_AB = !idx_mem_AB;
                if (patch_start_k+patch_w < k) {
                    VLMO_memcpy_patch(desc.device_A[idx_mem_AB], desc.host_A, patch_start_h, patch_h, patch_start_k+patch_w, patch_w, VLMO_Memcpy_HtoD, idx_mem_AB);
                    VLMO_memcpy_patch(desc.device_B[idx_mem_AB], desc.host_B, patch_start_k+patch_w, patch_h, patch_start_w, patch_w, VLMO_Memcpy_HtoD, idx_mem_AB);        
                }
                
            }

            /** Transfer data : stream #0 **/
            // Get result from device to host
            
            if (patch_start_h_pre != patch_start_h || patch_start_w_pre != patch_start_w) {
                VLMO_memcpy_patch(desc.host_C, desc.device_C[!idx_mem_C], patch_start_h_pre, patch_h, patch_start_w_pre, patch_w, VLMO_Memcpy_DtoH, !idx_mem_C);
            }
            
                            
            /** Synchronize device **/
            cudaErrChk (cudaStreamSynchronize (desc.streams[0]));
            cudaErrChk (cudaStreamSynchronize (desc.streams[1]));
            cudaErrChk (cudaGetLastError ());


            /** Update previous state **/
            patch_start_h_pre = patch_start_h;
            patch_start_w_pre = patch_start_w;

            printf("===========================================\n");
        }
    }
    // Get last result from device to host
    VLMO_memcpy_patch(desc.host_C, desc.device_C[idx_mem_C], patch_start_h_pre, patch_h, patch_start_w_pre, patch_w, VLMO_Memcpy_DtoH, idx_mem_C);


}


void VLMO_memcpy_patch(float* A, float* B, size_t patch_start_h, size_t patch_h, size_t patch_start_w, size_t patch_w, int mode, int idx_mem) {
    if (mode==VLMO_Memcpy_HtoD) {
        printf("Send [%lu %lu][%d]\n", patch_start_h, patch_start_w, idx_mem);
    } else {
        printf("Receive [%lu %lu][%d]\n", patch_start_h, patch_start_w, idx_mem);
    }
}



/******************************************************
  *****************************************************
  * Functions for matrix transposition
  *****************************************************
  *******************************************************/

void VLMO_matrix_transpose (VLMO_Operator_Descriptor_t& desc, const bool measure) {
    

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


