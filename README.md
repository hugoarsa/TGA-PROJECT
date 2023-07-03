# Proyecto final Tarjetas Gráficas

Este es el proyecto final desarrollado para la asignatura de tarjetas graficas y aceleradores desarollado por Hugo Aranda Sánchez y Paula Giner Muñoz 

### En que consiste

El proyecto consiste en una version paralela de un codigo de procesado de imagenes que aplica filtros diversos mediante un algoritmo de convolución génerico parametrizado para un mayor reaprovechamiento de codigo.

Este aplica diversos filtros descritos en la documentación sobre imagenes en blanco y negro o color. 

Utiliza memoria compartida para optimizar los tiempos y como mejoras futuras se planea añadir soporte para streaming pues creemos que dotaria el proyecto de una agilidad computacional adicional.

### Como ejecutar el contenido de las carpetas

Este proyecto ha sido creado y compilado en un sistema remoto de la universidad que ya tenía todos los prerequisitos instalados de CUDA

Para compilar se hace con: 
```make```
Y para ejecutarlo con una GPU se ha de usar:
```sbatch job.sh```

Si queremos observar como va la ejecución de forma dinamica (ver cuanto tiempo lleva, como va en la cola, si alguien la ocupa antes que nosotros) podemos usar:
```watch squeue```


