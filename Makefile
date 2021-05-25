CC = /usr/local/cuda/bin/nvcc
PROGS = 
CUDAFILES = src/core.cu src/operations.cu src/kernel.cu
INCS = include/core.cuh include/operations.cuh include/kernel.cuh
OPTIONS = -lcuda -O3

all: ${PROGS}

%: %.cu ${CUDAFILES} ${INCS} Makefile
	${CC} ${OPTIONS} -o $@ ${CUDAFILES} $<


clean :
	rm ${PROGS}



