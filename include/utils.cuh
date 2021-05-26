#ifndef __UTILS__
#define __UTILS__

void VLMO_record_start (cudaEvent_t& event_start, cudaEvent_t& event_end);
float VLMO_record_end (cudaEvent_t& event_start, cudaEvent_t& event_end);

#endif
