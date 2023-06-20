# Version 5 del kernel para Convolve génerico

Aqui podemos encontrar la quinta y última version del proyecto donde ahora el convolve 
también puede trabajar con imágenes en blanco y negro. Cosa que mejora algunos filtros 
como el de laplace haciendo que sea más facil ver los bordes.

### Como ejecutar

Se compila con un simple 
```make```

Y para ejecutarlo con una GPU se ha de usar
```sbatch job.sh```

Si queremos observar como va la ejecución de forma dinamica (ver cuanto tiempo lleva, como va en la cola, si alguien la ocupa antes que nosotros) podemos usar
```watch squeue```

### Cosas a destacar de la version

En esta version usamos un kernel especial para pasar la imagen a blanco y negro si se desea.

