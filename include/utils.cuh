#ifndef __UTILS__
#define __UTILS__

#include "core.cuh"

void VLMO_record_start (cudaEvent_t& event_start, cudaEvent_t& event_end);
float VLMO_record_end (cudaEvent_t& event_start, cudaEvent_t& event_end);
void get_maximum_size_patch (VLMO_Operator_Descriptor_t& desc);


#endif
