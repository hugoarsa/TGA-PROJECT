CUDA_HOME   = /Soft/cuda/11.2.1
CUDA_HOME   = /Soft/cuda/11.2.1

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include -arch=compute_35 --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
EXE	        = filtrarLP.exe
OBJ	        = filtrarLP.o

default: $(EXE)

filtrar.o: filtrar.cu
	$(NVCC) -c -o $@ filtrar_lp.cu  $(NVCC_FLAGS)  -I/Soft/stb/20200430  

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o *output.* FILTRARLP.e1* FILTRARLP.o1* *.o* *.e*

