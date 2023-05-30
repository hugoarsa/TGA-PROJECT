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


./filtrarLP.exe

./filtrarLP.exe default.bmp output.png


