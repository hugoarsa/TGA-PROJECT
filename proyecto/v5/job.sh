#!/bin/bash

### Directivas para el gestor de colas
#SBATCH --job-name=MultiGPU
#SBATCH -D .
#SBATCH --output=submit-FILTRO.o%j
#SBATCH --error=submit-FILTRO.e%j
#SBATCH -A cuda
#SBATCH -p cuda
### Se pide 1 GPU 
#SBATCH --gres=gpu:1

export PATH=/Soft/cuda/11.2.1/bin:$PATH


./filtrar.exe

./filtrar.exe default.bmp output0.png 0 0

./filtrar.exe default.bmp output1.png 1 0

./filtrar.exe default.bmp output2.png 2 0

./filtrar.exe default.bmp output3.png 3 0

./filtrar.exe default.bmp output4.png 4 0

./filtrar.exe default.bmp output5.png 0 1

./filtrar.exe default.bmp output6.png 1 1

./filtrar.exe default.bmp output7.png 2 1

./filtrar.exe default.bmp output8.png 3 1

./filtrar.exe default.bmp output9.png 4 1

