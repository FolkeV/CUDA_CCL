##############################################################
NVCC=nvcc
SRC = ./main.cu
CUDAFLAGS= -arch=sm_62
RM=/bin/rm -f

LDFLAGS += -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

main: main.o  CCL.o
	${NVCC} ${CUDAFLAGS} $(LDFLAGS) -o main main.o CCL.o

main.o: CCL.cuh utils.hpp main.cu
	$(NVCC) $(CUDAFLAGS) -std=c++11 -c main.cu

CCL.o: CCL.cuh CCL.cu reduction.cuh
	${NVCC} ${CUDAFLAGS} -std=c++11 -c CCL.cu

clean:
	${RM} *.o main