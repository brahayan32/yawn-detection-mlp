# Guia de exposicion

## Distribucion sugerida por integrante

### Integrante 1: Dataset y problema

Explicar:

- objetivo del proyecto,
- clases `yawn` y `no_yawn`,
- organizacion train/validation/test,
- conteo real de imagenes,
- limitacion de no llegar a 300 imagenes por clase.

Comando demo:

```powershell
py tools/dataset_summary.py
```

### Integrante 2: Preprocesamiento y modelo MLP

Explicar:

- escala de grises,
- recorte de rostro inferior/boca,
- GaussianBlur 3x3,
- resize `80x80`,
- normalizacion,
- flatten a `6400`,
- MLP `6400 -> 256 -> 64 -> 1`.

Respuesta clave:

```text
Usamos 80x80 porque el modelo entrenado espera 6400 caracteristicas.
```

### Integrante 3: OpenMP

Explicar:

- que OpenMP permite paralelizar bucles en CPU desde C,
- que se uso `#pragma omp parallel for`,
- que se compila con `gcc -fopenmp`,
- que se paralelizaron operaciones por pixel,
- que se comparo serial vs 1, 2, 4 y 8 hilos,
- que el speedup se calcula como tiempo serial dividido entre tiempo paralelo.

Comandos demo:

```powershell
cd parallel/openmp
mingw32-make
.\benchmark_openmp.exe 160 120 5
cd ..\..
py parallel/scripts/plot_openmp_speedup.py
```

### Integrante 4: CUDA

Explicar:

- CUDA paraleliza en GPU y se mantiene en archivos `.cu`,
- se implementaron kernels propios,
- el codigo usa estilo CUDA C con arreglos planos y `cudaMalloc/cudaMemcpy/cudaFree`,
- se respeto la arquitectura del MLP,
- se comparo forward CPU vs forward CUDA,
- el entrenamiento principal sigue en TensorFlow.

Kernels:

```text
matmul_kernel
bias_relu_kernel
sigmoid_kernel
binary_cross_entropy_kernel
```

Comandos demo:

```powershell
cd cuda
mingw32-make
.\benchmark_cuda.exe
cd ..
py cuda/plot_cuda_speedup.py
```

Resultado real para mencionar:

```text
batch 1: 24.585x
batch 8: 358.551x
batch 16: 398.811x
batch 32: 349.470x
```

Nota tecnica:

```text
En Windows, nvcc puede requerir cl.exe. En este proyecto la configuracion ya se resolvio y el benchmark CUDA se ejecuto correctamente.
```

### Integrante 5: Streamlit y resultados

Explicar:

- carga de imagen,
- captura por camara,
- camara en vivo con `streamlit-webrtc`,
- prediccion,
- probabilidad,
- uso del mismo preprocesamiento.

Comando demo:

```powershell
py -m streamlit run app_streamlit/streamlit_app.py
```

## Preguntas probables del profesor

### Por que no se uso 64x64 si la rubrica lo menciona?

Porque el modelo entrenado del proyecto usa entradas `80x80`, que al aplanarse producen `6400` caracteristicas. Cambiar a `64x64` produciria `4096` y no seria compatible con los pesos entrenados.

### Que parte paraleliza OpenMP?

Paraleliza bucles de pixeles en escala de grises, GaussianBlur, resize y normalizacion/flatten. Cada pixel de salida puede calcularse independientemente.

### Que significa speedup?

Es la relacion entre el tiempo base y el tiempo optimizado:

```text
speedup = tiempo_base / tiempo_optimizado
```

### Por que OpenMP con 1 hilo puede ser mas lento que serial?

Porque OpenMP agrega overhead de gestion de regiones paralelas. Con cargas pequenas, ese overhead puede superar el beneficio.

### Que implementa CUDA?

Implementa forward pass del MLP con kernels propios para multiplicacion de matrices, bias, ReLU, sigmoid y BCE. Tambien deja estructura inicial para gradientes y actualizacion SGD.

### CUDA reemplaza TensorFlow?

No. TensorFlow sigue siendo el entrenamiento principal. CUDA se implemento como evidencia academica paralela para demostrar kernels propios y benchmark CPU/GPU.

### Por que antes `cpu_loss_acc` y `cuda_loss` no coincidian?

Porque `cpu_loss_acc` acumulaba la perdida durante 5 repeticiones y `cuda_loss` estaba en escala promedio. El benchmark se ajusto para reportar `cpu_loss_avg` y `cuda_loss_avg` como valores comparables.

### Cual es la mayor limitacion del proyecto?

El dataset es pequeno y no cumple 300 imagenes por clase. Eso puede afectar generalizacion.

### Que metrica mirarian primero para seguridad?

Recall, porque interesa detectar la mayor cantidad posible de bostezos reales y evitar falsos negativos.

## Orden de demo recomendado

1. Mostrar dataset y conteos.
2. Mostrar app Streamlit.
3. Ejecutar benchmark OpenMP.
4. Mostrar grafica OpenMP.
5. Ejecutar o mostrar resultados del benchmark CUDA.
6. Mostrar reporte final y limitaciones.
