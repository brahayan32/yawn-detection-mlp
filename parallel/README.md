# Paralelizacion del proyecto

Esta carpeta contiene implementaciones auxiliares para cubrir la parte de programacion paralela de la rubrica del proyecto.

## Modulos

- `openmp/`: preprocesamiento de imagenes en C puro con version serial y version paralela usando OpenMP.
- `scripts/`: scripts de apoyo para graficar resultados de benchmarks.

## Relacion con el modelo principal

El modelo entrenado del proyecto recibe vectores de `6400` caracteristicas. Por eso el preprocesamiento paralelo usa imagenes normalizadas de `80x80`, ya que:

```text
80 x 80 = 6400
```

Aunque la rubrica mencione `64x64`, cambiar el tamano a `64x64` produciria vectores de `4096` caracteristicas y no seria compatible con el MLP entrenado.

## Evidencias esperadas

Despues de ejecutar el benchmark OpenMP se generan:

```text
metrics/openmp_benchmark.csv
metrics/openmp_speedup.png
```
