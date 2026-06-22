# Preprocesamiento OpenMP en C

Este modulo implementa en C puro una version serial y una version paralela del preprocesamiento usado por el proyecto de deteccion de bostezos.

## Operaciones implementadas

El pipeline implementado en C incluye:

1. Conversion RGB a escala de grises.
2. Filtro Gaussiano 3x3, alineado con `BLUR_KERNEL = (3, 3)` en `src/preprocessing.py`.
3. Redimensionamiento bilineal a `80x80`.
4. Normalizacion de pixeles a rango `0-1`.
5. Flatten a un vector de `6400` caracteristicas.

La version paralela usa OpenMP con directivas:

```c
#pragma omp parallel for
```

Estas directivas se aplican sobre bucles de pixeles porque cada posicion de salida puede calcularse de forma independiente.

## Por que 80x80 y no 64x64

La rubrica menciona `64x64`, pero el modelo actual del repositorio fue entrenado con entradas de `80x80`.

```text
80 x 80 = 6400 caracteristicas
```

Si se cambiara a `64x64`, la entrada seria:

```text
64 x 64 = 4096 caracteristicas
```

Eso no coincide con la arquitectura entrenada del MLP y romperia la inferencia actual.

## Compilacion

En Windows con MinGW:

```powershell
cd parallel/openmp
mingw32-make
```

El `Makefile` compila con `gcc` y `-fopenmp`.

En Linux/WSL:

```bash
cd parallel/openmp
make
```

## Ejecucion

```powershell
.\benchmark_openmp.exe
```

El benchmark tambien acepta argumentos opcionales:

```powershell
.\benchmark_openmp.exe 320 240 30
```

Donde:

- `320`: ancho sintetico de entrada.
- `240`: alto sintetico de entrada.
- `30`: repeticiones por imagen.

## Salida esperada

El benchmark genera:

```text
metrics/openmp_benchmark.csv
```

Con columnas:

```text
mode,threads,images,time_ms,speedup
```

## Grafica

Desde la raiz del proyecto:

```powershell
py parallel/scripts/plot_openmp_speedup.py
```

Esto genera:

```text
metrics/openmp_speedup.png
```

## Limitacion importante

Para evitar depender de OpenCV C++ o librerias externas de lectura JPEG/PNG, el benchmark usa una carga sintetica RGB deterministica con la misma cantidad de imagenes detectadas en `datasets/`.

El objetivo del benchmark es medir el costo computacional de las operaciones paralelizables del preprocesamiento, no reemplazar el pipeline Python/OpenCV usado por la app.
