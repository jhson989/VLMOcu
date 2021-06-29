
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

    if (desc.flag_cublas == true) {
        printf("[OPTIM] cuBLAS context created..\n");
        cuBLASErrChk (cublasCreate (&desc.handle));
    }



    if (desc.flag_unified_mem == true) {
        VLMO_matrix_multiplication_unified (desc);
    } else {
        VLMO_matrix_multiplication_patch (desc);
    }

    if (desc.flag_cublas == true) {
        cuBLASErrChk (cublasDestroy (desc.handle));
    }


    // Performance measurement
    if (measure == true) {
        VLMO_record_end (event_start, event_stop);
    }


}

void VLMO_matrix_multiplication_unified (VLMO_Operator_Descriptor_t& desc) {
    
    dim3 threads = desc.num_threads;
    dim3 blocks = dim3((desc.C_w+desc.num_threads.x-1) / desc.num_threads.x, (desc.C_h+desc.num_threads.y-1) / desc.num_threads.y);

    size_t m = desc.C_h;
    size_t n = desc.B_w;
    size_t k = desc.A_w;

    cuda_matrix_mul_basic<<<blocks, threads>>> (desc.device_A[0], desc.device_B[0], desc.device_C[0], m, n, k);
    cudaDeviceSynchronize(); 
    cudaErrChk( cudaGetLastError ());

}


void VLMO_matrix_multiplication_patch (VLMO_Operator_Descriptor_t& desc) {

    bool idx_mem_C=true, idx_mem_AB=true;
    size_t patch_h=desc.patch_h, patch_w=desc.patch_w, patch_k=desc.patch_w;
    size_t patch_start_h=0, patch_start_w=0, patch_start_k;
    size_t patch_start_h_pre=0, patch_start_w_pre=0;
    //size_t patch_start_h_next=0, patch_start_w_next=0;
    

    size_t m = desc.C_h;
    size_t n = desc.B_w;
    size_t k = desc.A_w;
    size_t len_k=0, len_w=0;

    // Send first input data from host to device
    cudaErrChk (cudaMemset (desc.device_C[0], 0, patch_w*patch_h*sizeof(float)));
    cudaErrChk (cudaMemset (desc.device_C[1], 0, patch_w*patch_h*sizeof(float)));
    for (patch_start_h=0; patch_start_h<desc.C_h; patch_start_h+=patch_h) {
        for (patch_start_w=0; patch_start_w<desc.C_w; patch_start_w+=patch_w) {         

            /** Update next state **/
            //if (patch_start_w + patch_w >= desc.C_w) patch_start_w_next = 0; else patch_start_w_next = patch_start_w + patch_w;
            //if (patch_start_w + patch_w >= desc.C_w) patch_start_h_next = patch_start_h + patch_h; else patch_start_h_next = patch_start_h;
            patch_start_k = 0;

            idx_mem_C = !idx_mem_C;
            //(int idx_mem, const size_t H_0, const size_t W_0, const size_t len, const size_t max_h, const size_t max_w)
            if (patch_start_k+patch_k>=k) len_k=k-(patch_start_k); else len_k=patch_k;
            VLMO_memcpy_patch(desc, desc.device_A[idx_mem_AB], desc.host_A, VLMO_Memcpy_HtoD, idx_mem_AB, patch_start_h, patch_start_k, len_k, desc.A_h, desc.A_w);
            if (patch_start_w+patch_w>=n) len_w=n-(patch_start_w); else len_w=patch_w;
            VLMO_memcpy_patch(desc, desc.device_B[idx_mem_AB], desc.host_B, VLMO_Memcpy_HtoD, idx_mem_AB, patch_start_k, patch_start_w, len_w, desc.B_h, desc.B_w);

            for (patch_start_k=0; patch_start_k<k; patch_start_k+=patch_k) {

                /** Synchronize device **/
                cudaErrChk (cudaStreamSynchronize (desc.streams[0]));
                cudaErrChk (cudaStreamSynchronize (desc.streams[1]));
                cudaErrChk (cudaGetLastError ());

                /** Launch kernel : stream #1 **/
                printf("    [Exec] patch [h%lu w%lu k%lu] data [AB%d] sum [C%d]\n", patch_start_h, patch_start_w, patch_start_k, (int)idx_mem_AB, (int)idx_mem_C);
                
                _VLMO_matrix_mul_patch (desc, desc.streams[1], desc.device_A[(int)idx_mem_AB], desc.device_B[(int)idx_mem_AB], desc.device_C[(int)idx_mem_C], m, n, k, patch_h, patch_w, patch_k, patch_start_h, patch_start_w, patch_start_k);    
                


                /** Send data from host to device **/
                idx_mem_AB = !idx_mem_AB;
                if (patch_start_k+patch_k < k) {
                    if (patch_start_k+2*patch_k>=k) len_k=k-(patch_start_k+patch_k); else len_k=patch_k;
                    VLMO_memcpy_patch(desc, desc.device_A[idx_mem_AB], desc.host_A, VLMO_Memcpy_HtoD, idx_mem_AB, patch_start_h, patch_start_k+patch_k, len_k, desc.A_h, desc.A_w);
                    VLMO_memcpy_patch(desc, desc.device_B[idx_mem_AB], desc.host_B, VLMO_Memcpy_HtoD, idx_mem_AB, patch_start_k+patch_k, patch_start_w, len_w, desc.B_h, desc.B_w);
                }
                
            }

            /** Transfer data : stream #0 **/
            // Get result from device to host
            if (patch_start_h_pre != patch_start_h || patch_start_w_pre != patch_start_w) {
                if (patch_start_w_pre+patch_w>=desc.C_w) len_w = desc.C_w-(patch_start_w_pre); else len_w = patch_w;
                VLMO_memcpy_patch(desc, desc.host_C, desc.device_C[!idx_mem_C], VLMO_Memcpy_DtoH, !idx_mem_C, patch_start_h_pre, patch_start_w_pre, len_w, desc.C_h, desc.C_w);
                cudaErrChk (cudaMemsetAsync (desc.device_C[!idx_mem_C], 0, patch_w*patch_h*sizeof(float), desc.streams[0]));        
            }
    
            /** Synchronize device **/
            cudaErrChk (cudaStreamSynchronize (desc.streams[0]));
            cudaErrChk (cudaStreamSynchronize (desc.streams[1]));
            cudaErrChk (cudaGetLastError ());


            /** Update previous state **/
            patch_start_h_pre = patch_start_h;
            patch_start_w_pre = patch_start_w;
        }
    }
    // Get last result from device to host
    if (patch_start_w_pre+patch_w>=desc.C_w) len_w = desc.C_w-(patch_start_w_pre); else len_w = patch_w;
    VLMO_memcpy_patch(desc, desc.host_C, desc.device_C[idx_mem_C], VLMO_Memcpy_DtoH, idx_mem_C, patch_start_h_pre, patch_start_w_pre, len_w, desc.C_h, desc.C_w); 
    cudaErrChk (cudaDeviceSynchronize ());
}


void VLMO_memcpy_patch(VLMO_Operator_Descriptor_t& desc, float* A, float* B, int mode, int idx_mem, const size_t H_0, const size_t W_0, const size_t len, const size_t max_h, const size_t max_w) {

    // TODO
    size_t H = desc.patch_h;
    if (H_0+H>max_h) H = max_h-H_0;
    if (mode == VLMO_Memcpy_HtoD) {
        for (size_t h=0; h<H; h++) {
            cudaErrChk (cudaMemcpyAsync (&A[h*desc.patch_w], &B[(H_0+h)*max_w+W_0], len*sizeof (float), cudaMemcpyHostToDevice, desc.streams[0]));
        }  
    } else {
        for (size_t h=0; h<H; h++) {
            cudaErrChk (cudaMemcpyAsync (&A[(H_0+h)*max_w+W_0], &B[h*desc.patch_w], len*sizeof (float), cudaMemcpyDeviceToHost, desc.streams[0]));
        }  
    }
}
void _VLMO_matrix_mul_patch (VLMO_Operator_Descriptor_t& desc, cudaStream_t& stream, const float *A, const float *B, float *C, const int M, const int N, const int K, const int patch_h, const int patch_w, const int patch_k, const int patch_start_h, const int patch_start_w, const int patch_start_k) {
    
        
    
    

    int remain_h = (M-patch_start_h) >= patch_h ? patch_h : (M-patch_start_h);
    int remain_w = (N-patch_start_w) >= patch_w ? patch_w : (N-patch_start_w);
    int remain_k = (K-patch_start_k) >= patch_k ? patch_k : (K-patch_start_k);
    
    // Fully loaded
    if ((M-patch_start_h) >= patch_h && (N-patch_start_w) >= patch_w && (K-patch_start_k) >= patch_k) {
        if (desc.flag_cublas == true) {
            float alpha = 1.0f, beta = 1.0f;
            cuBLASErrChk (cublasSgemm (desc.handle, CUBLAS_OP_N, CUBLAS_OP_N, remain_w, remain_h, remain_k, &alpha, B, remain_w, A, remain_k, &beta, C, remain_w ) );
        } else {
            dim3 threads = dim3(32, 32);
            //dim3 blocks = dim3(  (((desc.patch_w+threads.x-1) / threads.x)+0 )/1, ( ((desc.patch_h+threads.y-1) / threads.y)+0 )/1  );
            dim3 blocks = dim3( (desc.patch_w+threads.x-1) / threads.x, (desc.patch_h+threads.y-1) / threads.y );
            const size_t size_smem = 2*sizeof(float)*threads.x*threads.x;
            cuda_matrix_mul_patch_tiled_full_loaded<1024*16, 32> <<<blocks, threads, size_smem, stream>>> (A, B, C, remain_h, remain_w, patch_k, patch_w);
        }
    }
    // Partially loaded
    else {
        dim3 threads = desc.num_threads;
        dim3 blocks = dim3((desc.patch_w+desc.num_threads.x-1) / desc.num_threads.x, (desc.patch_h+desc.num_threads.y-1) / desc.num_threads.y);
        const size_t size_smem = 2*sizeof(float)*threads.x*threads.x;
        cuda_matrix_mul_patch_tiled <<<blocks, threads, size_smem, stream>>> (A, B, C, remain_h, remain_w, remain_k, patch_k, patch_w);
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



