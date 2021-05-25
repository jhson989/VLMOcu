CC = /usr/local/cuda/bin/nvcc
PROGS = 
LIBS = src/core.cu src/operations.cu
INCS = include/core.cuh include/operations.cuh
OPTIONS = -lcuda -O3

all: ${PROGS}

%: %.cu ${LIBS} ${INCS} Makefile
	${CC} ${OPTIONS} -o $@ ${LIBS} $<


clean :
	rm ${PROGS}



