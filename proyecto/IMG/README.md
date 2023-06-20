
# Version iterativa para Gauss-Seidel

Aqui podemos encontrar la version iterativa donde trabajamos el kernel de Gauss-Seidel 
(como kernel generico pues luego lo modificaremos para admitir otros filtros) esta tiene
como funcion unicamente entender como se importan y exporan imagenes para el formato dado 

### Como ejecutar

Se compila con un simple 
```make```
Y para ejecutarlo con una CPU de forma no interactiva se ha de usar
```sbatch job.sh```

De forma interactiva basta con ejecutarlo tal cual especificando imagen de entrada y salida
```./filtrar.exe <imagen entrada> <imagen salida>```

### Cosas a destacar de la version

Esta version tiene como unica funcion comenzar a tontear con el algoritmo a paralelizar y 
entender bien como funciona la entrada y salida de imagenes de diferentes formatos.

También por lo tanto ver las compatibilidades que tiene la libreria dada evaluando que
formatos sabe procesar y cuales no.
