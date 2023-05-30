#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#ifndef SIZE
#define SIZE 32
#endif

#ifndef PINNED
#define PINNED 0
#endif


// kernel elemento a elemento para el filtro Laplace

__global__ void Laplace_kernel(int N, int M, unsigned char *source, unsigned char *dest) {
  int Kernel[] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
  int kernelSize = 3;

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N && j < M) {
    int sum = 0;

    for (int k = 0; k < kernelSize; ++k) {
      for (int l = 0; l < kernelSize; ++l) {
        int y = i + k - kernelSize / 2;
        int x = j + l - kernelSize / 2;

        if (y >= 0 && y < N && x >= 0 && x < M) {
          sum += source[y * M + x] * Kernel[k * kernelSize + l];
        }
      }
    }

    int offset = i * M + j;
    dest[offset] = abs(sum);
  }
}


void InitM(int N, int M, float *Mat);


int main(int argc, char **argv) {
  unsigned int numBytesI;
  unsigned int nBlocksW, nBlocksH, nThreads;

  float TiempoTotal, TiempoKernel;
  cudaEvent_t E0, E1, E2, E3;

  unsigned char *h_output;
  unsigned char *d_image, *d_output;

  // Ficheros de entrada y de salida
  char *fileIN, *fileOUT;

  if (argc == 3) {
    fileIN = argv[1];
    fileOUT = argv[2];
  } else {
    printf("Usage: ./exe fileIN fileOUT\n");
    exit(0);
  }


  // Lectura de imagenes
  unsigned char *h_image;
  // Meta info de la imagen
  int width, height, pixelWidth;

  printf("Reading image...\n");
  h_image = stbi_load(fileIN, &width, &height, &pixelWidth, 0);

  if (!h_image) {
    fprintf(stderr, "Couldn't load image.\n");
    return (-1);
  }
  printf("Image Read. Width: %d, Height: %d, nComp: %d\n", width, height, pixelWidth);


  // Buscar GPU de forma aleatoria
  int count, gpu;
  cudaGetDeviceCount(&count);
  srand(time(NULL));
  gpu = (rand() >> 3) % count;
  cudaSetDevice(gpu);


  // Numero de Threads en cada dimension
  nThreads = SIZE;


  // Numero de Blocks en cada dimension
  nBlocksW = (width + nThreads - 1) / nThreads;
  nBlocksH = (height + nThreads - 1) / nThreads;


  // Calculamos la memoria a reservar
  numBytesI = width * height * 3 * sizeof(unsigned char);


  // Establecemos las dimensiones
  dim3 dimGrid(nBlocksW, nBlocksH, 1);
  dim3 dimBlock(nThreads, nThreads, 1);


  cudaEventCreate(&E0);
  cudaEventCreate(&E1);
  cudaEventCreate(&E2);
  cudaEventCreate(&E3);


  if (PINNED) {
    // Obtiene Memoria [pinned] en el host
    cudaMallocHost((unsigned char **)&h_output, numBytesI);
  } else {
    // Obtener Memoria en el host
    h_output = (unsigned char *)malloc(numBytesI);
  }

  cudaEventRecord(E0, 0);
  cudaEventSynchronize(E0);

  // Obtener Memoria en el device
  cudaMalloc((unsigned char **)&d_image, numBytesI);
  cudaMalloc((unsigned char **)&d_output, numBytesI);

  // Copiar datos desde el host en el device
  cudaMemcpy(d_image, h_image, numBytesI, cudaMemcpyHostToDevice);

  cudaEventRecord(E1, 0);
  cudaEventSynchronize(E1);

  // Ejecutar el kernel
  Laplace_kernel<<<dimGrid, dimBlock>>>(height, width, d_image, d_output);

  cudaEventRecord(E2, 0);
  cudaEventSynchronize(E2);

  // Obtener el resultado desde el host
  cudaMemcpy(h_output, d_output, numBytesI, cudaMemcpyDeviceToHost);

  // Liberar Memoria del device
  cudaFree(d_output);
  cudaFree(d_image);

  cudaEventRecord(E3, 0);
  cudaEventSynchronize(E3);

  cudaEventElapsedTime(&TiempoTotal, E0, E3);
  cudaEventElapsedTime(&TiempoKernel, E1, E2);
  printf("\nKERNEL 01\n");
  printf("GPU utilizada: %d\n", gpu);
  printf("Dimensiones: height = %d, width = %d\n", height, width);
  printf("nThreads: %dx%d (%d)\n", nThreads, nThreads, nThreads * nThreads);
  printf("nBlocks: %dx%d (%d)\n", nBlocksW, nBlocksH, nBlocksW * nBlocksH);
  if (PINNED) printf("Usando Pinned Memory\n");
         else printf("NO usa Pinned Memory\n");
  printf("Tiempo Global: %4.6f milseg\n", TiempoTotal);
  printf("Tiempo Kernel: %4.6f milseg\n", TiempoKernel);
  //printf("Rendimiento Global: %4.2f GFLOPS\n", (2.0 * (float) N * (float) M * (float) P) / (1000000.0 * TiempoTotal));
  //printf("Rendimiento Kernel: %4.2f GFLOPS\n", (2.0 * (float) N * (float) M * (float) P) / (1000000.0 * TiempoKernel));

  cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);

  printf("Escribiendo\n");
  //ESCRITURA DE LA IMAGEN EN SECUENCIAL
  stbi_write_png(fileOUT,width,height,pixelWidth,h_output,0);

  if (PINNED) {
    cudaFreeHost(h_output);
  }
  else {
    free(h_output);
  }

}