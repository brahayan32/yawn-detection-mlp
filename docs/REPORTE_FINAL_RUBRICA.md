# Reporte final segun rubrica

## 1. Resumen del proyecto

`yawn-detection-mlp` es un proyecto de vision artificial para detectar si una persona esta bostezando o no. El sistema trabaja con dos clases:

```text
yawn
no_yawn
```

El modelo principal es un MLP implementado en TensorFlow. La entrada del modelo tiene `6400` caracteristicas porque cada imagen se procesa a `80x80` y luego se aplana:

```text
80 x 80 = 6400
```

## 2. Dataset

El dataset esta organizado en:

```text
datasets/train/no_yawn
datasets/train/yawn
datasets/validation/no_yawn
datasets/validation/yawn
datasets/test/no_yawn
datasets/test/yawn
```

Conteo real generado con:

```powershell
py tools/dataset_summary.py
```

Resumen:

| Split | no_yawn | yawn | Total |
|---|---:|---:|---:|
| train | 88 | 114 | 202 |
| validation | 26 | 27 | 53 |
| test | 27 | 26 | 53 |
| **Total** | **141** | **167** | **308** |

Evidencias:

```text
metrics/dataset_summary.csv
metrics/dataset_distribution.png
docs/DATASET_ANALYSIS.md
```

Limitacion: el proyecto no cumple el minimo ideal de 300 imagenes por clase. Esta es una limitacion real, no un resultado inventado.

## 3. Preprocesamiento Python/OpenCV

El preprocesamiento principal esta en:

```text
src/preprocessing.py
app_streamlit/utils.py
```

Pipeline:

1. Lectura de imagen.
2. Conversion a escala de grises.
3. Recorte `lower_face`/boca.
4. `GaussianBlur` con `BLUR_KERNEL = (3, 3)`.
5. Resize a `80x80`.
6. Normalizacion a rango `0-1`.
7. Flatten a `6400`.

Fragmento clave:

```python
blurred = cv2.GaussianBlur(gray_crop, BLUR_KERNEL, 0)
resized = cv2.resize(blurred, IMAGE_SIZE)
normalized = resized.astype("float32") / 255.0
input_vector = normalized.flatten().reshape(1, -1)
```

## 4. Preprocesamiento OpenMP

La implementacion OpenMP esta en:

```text
parallel/openmp/openmp_preprocessing.c
parallel/openmp/benchmark_openmp.c
parallel/openmp/README.md
```

Incluye version serial y version paralela para:

- conversion RGB a escala de grises,
- filtro Gaussiano 3x3,
- resize bilineal a `80x80`,
- normalizacion,
- flatten a `6400`.

Se usa:

```c
#pragma omp parallel for
```

Esta parte esta implementada en C puro y se compila con `gcc -fopenmp`.

Las operaciones por pixel son paralelizables porque cada pixel de salida puede calcularse de forma independiente.

Aunque la rubrica menciona `64x64`, este proyecto usa `80x80` porque el MLP entrenado espera `6400` caracteristicas. Usar `64x64` produciria `4096` caracteristicas y romperia la compatibilidad con el modelo actual.

## 5. Modelo MLP

El modelo principal esta en:

```text
src/model.py
```

Arquitectura:

```text
Entrada: 6400
Capa oculta 1: 256, ReLU
Capa oculta 2: 64, ReLU
Salida: 1, Sigmoid
```

Modelos exportados:

```text
models/best_model
models/final_model
```

La app intenta cargar primero `models/best_model` y, si falla, `models/final_model`.

## 6. Implementacion CUDA

La implementacion CUDA C esta en:

```text
cuda/kernels.cuh
cuda/mlp_cuda.cu
cuda/benchmark_cuda.cu
cuda/README.md
```

Kernels implementados:

- `matmul_kernel`
- `bias_relu_kernel`
- `bias_kernel`
- `sigmoid_kernel`
- `binary_cross_entropy_kernel`
- `output_gradient_kernel`
- `sgd_update_kernel`

Alcance:

```text
forward pass CUDA completo + BCE CUDA + benchmark CPU/GPU
```

El entrenamiento principal sigue con TensorFlow. CUDA se implementa como evidencia academica de paralelizacion del modelo y comparacion de rendimiento.

## 7. Benchmarks y speedup

OpenMP:

```text
metrics/openmp_benchmark.csv
metrics/openmp_speedup.png
```

Resultados reales del benchmark OpenMP:

| Modo | Hilos | Imagenes procesadas | Tiempo ms | Speedup |
|---|---:|---:|---:|---:|
| serial | 1 | 616 | 5008.000 | 1.000 |
| openmp | 1 | 616 | 4597.000 | 1.089 |
| openmp | 2 | 616 | 2132.000 | 2.349 |
| openmp | 4 | 616 | 1716.000 | 2.918 |
| openmp | 8 | 616 | 1575.000 | 3.180 |

CUDA:

```text
metrics/cuda_benchmark.csv
metrics/cuda_speedup.png
```

Resultados reales del benchmark CUDA:

| Batch size | CPU ms | CUDA ms | Speedup |
|---:|---:|---:|---:|
| 1 | 64.000 | 2.603 | 24.585 |
| 8 | 467.000 | 1.302 | 358.551 |
| 16 | 838.000 | 2.101 | 398.811 |
| 32 | 1570.000 | 4.493 | 349.470 |

El benchmark CUDA compilo y ejecuto correctamente, generando CSV y grafica reales.

Nota sobre loss del benchmark CUDA: en una salida anterior, `cpu_loss_acc` era la suma acumulada de la BCE durante 5 repeticiones, mientras que `cuda_loss` era un promedio de BCE. Por eso no estaban en la misma escala. El benchmark fue ajustado para reportar `cpu_loss_avg` y `cuda_loss_avg` en futuras ejecuciones, de modo que ambos valores sean comparables.

Formula de speedup:

```text
speedup = tiempo_base / tiempo_paralelo
```

Para OpenMP, la base es el tiempo serial. Para CUDA, la base es el tiempo CPU.

## 8. Metricas y analisis

Metricas principales:

| Metrica | Valor |
|---|---:|
| Accuracy | 0.8214 |
| Precision | 0.8462 |
| Recall | 0.7857 |
| F1-score | 0.8148 |

Evidencias:

```text
metrics/results.csv
metrics/confusion_matrix.png
metrics/accuracy.png
metrics/loss.png
docs/ANALISIS_METRICAS.md
```

## 9. App Streamlit

La app esta en:

```text
app_streamlit/streamlit_app.py
app_streamlit/utils.py
app_streamlit/README.md
```

Permite:

- subir imagen,
- capturar foto,
- usar camara en vivo si `streamlit-webrtc` esta instalado,
- cargar el modelo entrenado,
- aplicar el preprocesamiento del proyecto,
- mostrar prediccion,
- mostrar probabilidad.

Comando:

```powershell
py -m streamlit run app_streamlit/streamlit_app.py
```

## 10. Evidencias

Evidencias generadas o referenciadas:

```text
metrics/dataset_summary.csv
metrics/dataset_distribution.png
metrics/openmp_benchmark.csv
metrics/openmp_speedup.png
metrics/results.csv
metrics/confusion_matrix.png
metrics/accuracy.png
metrics/loss.png
```

Evidencias CUDA generadas:

```text
metrics/cuda_benchmark.csv
metrics/cuda_speedup.png
```

## 11. Limitaciones

- Dataset pequeno y sin 300 imagenes por clase.
- No hay documento formal de origen del dataset en el repositorio.
- El benchmark OpenMP usa carga RGB sintetica deterministica para evitar dependencia de OpenCV C++.
- CUDA en Windows puede requerir `cl.exe` de Visual Studio Build Tools. En este proyecto esa configuracion ya fue resuelta para ejecutar el benchmark real.
- La implementacion CUDA no reemplaza el entrenamiento TensorFlow.
- El MLP no aprovecha la estructura espacial como una CNN.

## 12. Conclusiones

El proyecto queda alineado con la rubrica en dataset, preprocesamiento, OpenMP, CUDA, metricas, app, evidencias y documentacion. OpenMP y CUDA cuentan con CSV y graficas generadas a partir de benchmarks reales.

## 13. Mejoras futuras

- Aumentar el dataset hasta al menos 300 imagenes por clase.
- Documentar formalmente el origen del dataset.
- Integrar OpenMP con lectura real de imagenes usando una biblioteca C compatible o OpenCV C++ si la rubrica lo permite.
- Completar entrenamiento CUDA con backpropagation total.
- Comparar MLP con CNN.
- Mejorar recall para reducir falsos negativos.
