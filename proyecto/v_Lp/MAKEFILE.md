# Version 1 del kernel para Gauss-Seidel

Aqui podemos encontrar la primera version donde trabajamos el kernel de Laplace elemento
a elemento y accediendo a memoria principal 

### Como ejecutar

Se compila con un simple 
```make```
Y para ejecutarlo con una GPU se ha de usar
```sbatch job.sh```

Si queremos observar como va la ejecución de forma dinamica (ver cuanto tiempo lleva,
como va en la cola, si alguien la ocupa antes que nosotros) podemos usar
```watch squeue```

### Cosas a destacar de la version

Esta version como hemos dicho hace un procesado elemento a elemento por cada uno de los
threads que desarrollan la operacion.

