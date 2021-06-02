
#include "../include/utils.cuh"

void VLMO_record_start (cudaEvent_t& event_start, cudaEvent_t& event_end) {

    cudaErrChk(cudaEventCreate(&event_start));
    cudaErrChk(cudaEventCreate(&event_end));
    cudaErrChk(cudaEventRecord(event_start, NULL));

}


float VLMO_record_end (cudaEvent_t& event_start, cudaEvent_t& event_end) {

    cudaErrChk(cudaEventRecord(event_end, NULL));
    cudaErrChk(cudaEventSynchronize(event_end));

    float msec = 0.0f;
    cudaErrChk(cudaEventElapsedTime(&msec, event_start, event_end));
    printf("[Perf] Elaped time: %.4f sec\n", msec*1e-3);

    return msec;
}

void get_maximum_size_patch (VLMO_Operator_Descriptor_t& desc) {
    
    
    desc.patch_w = 15;
    desc.patch_h = 15;

}

