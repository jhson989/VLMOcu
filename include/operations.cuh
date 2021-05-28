#ifndef __OPERATIONS__
#define __OPERATIONS__

#include "core.cuh"

/** Matrix element-wise operations **/
void VLMO_element_operation (VLMO_Operator_Descriptor_t& desc, const bool measure);
void VLMO_element_unified (VLMO_Operator_Descriptor_t& desc);
void VLMO_element_patch (VLMO_Operator_Descriptor_t& desc);


/** Matrix multiplication **/
void VLMO_matrix_multiplication (VLMO_Operator_Descriptor_t& desc, VLMO_Operator_t, const bool measure);
void VLMO_matrix_multiplication_unified (VLMO_Operator_Descriptor_t& desc);

void VLMO_matrix_transpose (VLMO_Operator_Descriptor_t& desc, VLMO_Operator_t, const bool measure);
void VLMO_matrix_transpose_unified (VLMO_Operator_Descriptor_t& desc);






#endif
