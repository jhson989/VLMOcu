#include "../include/core.cuh"


int main(void) {

    int device_num = VLMO_get_device_num(true);
 
    for (int i=0; i<device_num; i++){
        size_t free, total;
        VLMO_get_device_properties (i, free, total, true);
    }

    return 0;
}

