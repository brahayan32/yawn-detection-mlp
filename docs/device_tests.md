# Pruebas de CPU y GPU

El MLP no realiza paralelismo por si mismo. El paralelismo ocurre cuando TensorFlow ejecuta operaciones numericas como multiplicaciones de matrices y calculo de gradientes.

## Comprobar dispositivos

Desde la raiz del proyecto:

```bash
python -m src.device_test --device both
```

La salida muestra las CPU y GPU detectadas por TensorFlow. Si aparece una GPU, TensorFlow puede usar CUDA para acelerar operaciones compatibles.

## Instalacion en WSL2 con GPU

Si `nvidia-smi` detecta la GPU pero TensorFlow no, instala TensorFlow con soporte CUDA dentro del entorno:

```bash
pip install -r requirements-gpu-wsl.txt
```

Esto usa `tensorflow[and-cuda]`, que incluye las librerias CUDA/cuDNN necesarias para TensorFlow en Linux/WSL2.

## Ejecutar solo CPU

```bash
python -m src.device_test --device cpu
```

## Ejecutar solo GPU

```bash
python -m src.device_test --device gpu
```

## Importante

Esta prueba usa datos sinteticos y pocos pasos. Sirve para verificar disponibilidad y comparar tiempos de ejecucion, pero no reemplaza el entrenamiento real ni las metricas finales del modelo.
