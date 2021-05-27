#ifndef __OPERATIONS__
#define __OPERATIONS__

#include "core.cuh"

/** Matrix element-wise operations **/
void VLMO_element_operation (VLMO_Operator_Descriptor_t& desc, VLMO_Operator_t, const bool measure);

void VLMO_element_addition (VLMO_Operator_Descriptor_t& desc);
void VLMO_element_addition_unified (VLMO_Operator_Descriptor_t& desc);

void VLMO_element_subtraction (VLMO_Operator_Descriptor_t& desc);
void VLMO_element_subtraction_unified (VLMO_Operator_Descriptor_t& desc);

void VLMO_element_multiplication (VLMO_Operator_Descriptor_t& desc);
void VLMO_element_multiplication_unified (VLMO_Operator_Descriptor_t& desc);

void VLMO_element_division (VLMO_Operator_Descriptor_t& desc);
void VLMO_element_division_unified (VLMO_Operator_Descriptor_t& desc);


/** Matrix multiplication **/
void VLMO_matrix_multiplication (VLMO_Operator_Descriptor_t& desc, VLMO_Operator_t, const bool measure);
void VLMO_matrix_multiplication_unified (VLMO_Operator_Descriptor_t& desc);

void VLMO_transposition (VLMO_Operator_Descriptor_t& desc);
void VLMO_transposition_unified (VLMO_Operator_Descriptor_t& desc);






#endif
