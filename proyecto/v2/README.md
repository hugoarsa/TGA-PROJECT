# Version 2 del kernel para Gauss-Seidel

Aqui podemos encontrar la segunda version donde trabajamos el kernel de Gauss-Seidel (como kernel generico pues luego lo modificaremos para admitir otros filtros) esta vez con el uso de shared memory

### Como ejecutar

Se compila con un simple 
```make```
Y para ejecutarlo con una GPU se ha de usar
```sbatch job.sh```

Si queremos observar como va la ejecución de forma dinamica (ver cuanto tiempo lleva, como va en la cola, si alguien la ocupa antes que nosotros) podemos usar
```watch squeue```

### Cosas a destacar de la version

Esta version como hemos dicho usa shared memory para los pixeles de dentro del block.

Una posible mejora de cara al futuro seria hacer que no solo los pixeles de dentro del 
block esten dentro sino los colindantes también. Pues en casos limite hemos de acceder
a estos para hacer las medias de los pixeles que estan en el borde y actualmente esto 
se hace con un acceso a memoria principal en vez de al shared.


