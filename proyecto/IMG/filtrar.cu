#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

char *fileIN, *fileOUT;
unsigned char *image;
int width, height, pixelWidth; //meta info de la imagen


int main(int argc, char** argv)
{
        // Ficheros de entrada y de salida
        if (argc == 3) { fileIN = argv[1]; fileOUT = argv[2]; }
        else { printf("Usage: ./exe fileIN fileOUT\n"); exit(0); }


        //lectura de la imagen
        printf("Reading image...\n");
        image = stbi_load(fileIN, &width, &height, &pixelWidth, 0);
        if (!image) {
                fprintf(stderr, "Couldn't load image.\n");
                return (-1);
        }
        printf("Image Read. Width : %d, Height : %d, nComp: %d\n",width,height,pixelWidth);

        float kernel[] = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};
        int kernelSize = 3;

	unsigned char *h_output;
	h_output = (unsigned char*) malloc(width*height*3*sizeof(unsigned char));

        printf("Filtrando\n");

        printf("Filtrando la imagen\n");

        for (int offset = 0; offset < width*height*3; offset+=3){ //simplified for
                int j = offset % (width*3);
                int i = (offset - j) / (width*3);

                float red = 0.0;
                float green = 0.0;
                float blue = 0.0;
                for (int k = 0; k < kernelSize; ++k){
                        for (int l = 0; l < kernelSize; ++l){
                        int y = i + k - kernelSize / 2;
                        int x = j + (l - kernelSize / 2)*3;
                        	if (y >= 0 && y < height && x >= 0 && x < width*3){
                                	red += image[y*width*3 + x] * kernel[k*3 + l];
                                	green += image[y*width*3 + x + 1] * kernel[k*3 + l];
                                	blue += image[y*width*3 + x + 2] * kernel[k*3 + l];
                        	}
                	}
                }
                h_output[offset] = red;
                h_output[offset+1] = green;
                h_output[offset+2] = blue;
        }

        printf("Escribiendo\n");
        //ESCRITURA DE LA IMAGEN EN SECUENCIAL
        stbi_write_png(fileOUT,width,height,pixelWidth,h_output,0);

	free(h_output);
}
