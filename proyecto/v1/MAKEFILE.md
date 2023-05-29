# Version 1 del kernel para Gauss-Seidel

Aqui podemos encontrar la primera version donde trabajamos el kernel de Gauss-Seidel
(como kernel generico pues luego lo modificaremos para admitir otros filtros) elemento
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

Una posible mejora de cara a proximas versiones seria el uso de memoria compartida para
hacer las lecturas a nivel de bloque. Otra posible mejora seria que cada thread tratase
una fila, pues tal vez perdamos menos tiempo en overheads.
