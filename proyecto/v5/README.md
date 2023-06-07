# Version 2 del kernel para Gauss-Seidel

Aqui podemos encontrar la tercera version donde trabajamos diversos kernels
mediante el uso de flags, haciendo uso también de shared memory

### Como ejecutar

Se compila con un simple 
```make```
Y para ejecutarlo con una GPU se ha de usar
```sbatch job.sh```

Si queremos observar como va la ejecución de forma dinamica (ver cuanto tiempo lleva, como va en la cola, si alguien la ocupa antes que nosotros) podemos usar
```watch squeue```

### Cosas a destacar de la version

Esta version como hemos dicho usa shared memory para los pixeles de dentro del block.

Además hace uso de diversos kernels diferentes para mostrar lo versatil que es el cuda
kernel de convolve_RGB el cual puede aplicar distintos kernels de procesado de imagen
obteniendo resultados muy diversos sin problemas.


