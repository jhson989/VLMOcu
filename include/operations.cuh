#ifndef __OPERATIONS__
#define __OPERATIONS__

#include "core.cuh"


void VLMO_addition (VLMO_Operator_Descriptor_t& desc);
void VLMO_addition_unified (VLMO_Operator_Descriptor_t& desc);

void VLMO_subtraction (VLMO_Operator_Descriptor_t& desc);
void VLMO_subtraction_unified (VLMO_Operator_Descriptor_t& desc);

void VLMO_multiplication (VLMO_Operator_Descriptor_t& desc);
void VLMO_multiplication_unified (VLMO_Operator_Descriptor_t& desc);

void VLMO_transposition (VLMO_Operator_Descriptor_t& desc);
void VLMO_transposition_unified (VLMO_Operator_Descriptor_t& desc);






#endif