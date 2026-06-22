# Implementacion CUDA C del MLP

Esta carpeta contiene una implementacion academica en CUDA C para evidenciar paralelizacion del modelo MLP usado en `yawn-detection-mlp`.

## Arquitectura respetada

```text
Entrada: 6400
Capa oculta 1: 256, ReLU
Capa oculta 2: 64, ReLU
Salida: 1, Sigmoid
```

La entrada `6400` corresponde al preprocesamiento `80x80`.

## Kernels implementados

- `matmul_kernel`: multiplicacion matriz-matriz para cada capa densa.
- `bias_relu_kernel`: suma de bias y activacion ReLU para capas ocultas.
- `bias_kernel`: suma de bias para la capa de salida.
- `sigmoid_kernel`: activacion sigmoid para probabilidad final.
- `binary_cross_entropy_kernel`: calculo de perdida BCE por muestra.
- `output_gradient_kernel`: gradiente de salida para estructura de backpropagation.
- `sgd_update_kernel`: actualizacion SGD generica para estructura de entrenamiento.

## Alcance

El benchmark implementa:

```text
forward pass CUDA completo + BCE CUDA + comparacion CPU/GPU
```

El entrenamiento principal del proyecto sigue estando en TensorFlow. La parte CUDA funciona como evidencia academica de kernels propios y comparacion de rendimiento.

La estructura de backpropagation queda iniciada con kernels de gradiente de salida y actualizacion SGD, pero no reemplaza al entrenamiento TensorFlow.

## Compilacion

En Windows:

```powershell
cd cuda
mingw32-make
```

En Windows, `nvcc` usa un compilador host como `cl.exe`. En este proyecto esa configuracion ya quedo resuelta y el benchmark compilo correctamente.

Aunque el codigo evita C++ innecesario y usa estilo CUDA C con arreglos planos, `malloc/free`, `printf`, `cudaMalloc`, `cudaMemcpy` y `cudaFree`, otros equipos pueden necesitar Visual Studio Build Tools o una terminal configurada con `cl.exe`.

En Linux/WSL:

```bash
cd cuda
make
```

## Ejecucion

```powershell
.\benchmark_cuda.exe
```

El benchmark genera:

```text
metrics/cuda_benchmark.csv
```

Con columnas:

```text
version,batch_size,repetitions,time_ms,speedup
```

## Resultado real obtenido

El benchmark CUDA ya compilo y se ejecuto correctamente en el entorno del proyecto. Resultados reales registrados en `metrics/cuda_benchmark.csv`:

| Batch size | CPU ms | CUDA ms | Speedup |
|---:|---:|---:|---:|
| 1 | 64.000 | 2.603 | 24.585 |
| 8 | 467.000 | 1.302 | 358.551 |
| 16 | 838.000 | 2.101 | 398.811 |
| 32 | 1570.000 | 4.493 | 349.470 |

Tambien se genero:

```text
metrics/cuda_speedup.png
```

## Nota sobre loss CPU/GPU

En una version anterior de la salida, `cpu_loss_acc` representaba la suma acumulada de la BCE durante `5` repeticiones, mientras que `cuda_loss` representaba el promedio de BCE de la ultima pasada. Por eso aparecian escalas distintas:

```text
cpu_loss_acc ~= 5 * cuda_loss
```

El benchmark fue ajustado para imprimir valores comparables:

```text
cpu_loss_avg
cuda_loss_avg
```

Los CSV de benchmark conservan tiempos y speedup; no incluyen loss.

## Grafica

Desde la raiz del proyecto:

```powershell
py cuda/plot_cuda_speedup.py
```

Esto genera:

```text
metrics/cuda_speedup.png
```

## Limitaciones

- En Windows, `nvcc` puede requerir `cl.exe` aunque el codigo use estilo CUDA C. En este proyecto esa configuracion ya fue resuelta para ejecutar el benchmark.
- El benchmark usa pesos deterministas sinteticos para comparar CPU/GPU sin modificar los modelos entrenados.
- No se reentrena el modelo principal.
- El speedup depende del tamano de batch, GPU, compilador y overhead de lanzamiento de kernels.
