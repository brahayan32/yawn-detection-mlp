# Entrenamiento

El modelo principal es un MLP, o perceptron multicapa, usado para clasificacion binaria supervisada. Esta implementacion usa TensorFlow puro, sin Keras.

## Arquitectura

```text
Entrada: 4096 caracteristicas
Capa oculta 1: 512 neuronas, ReLU
Capa oculta 2: 128 neuronas, ReLU
Salida: 1 neurona, Sigmoid
```

## Implementacion sin Keras

El modelo se define con `tf.Variable` para pesos y sesgos. La propagacion hacia adelante usa operaciones de TensorFlow:

- `tf.matmul` para multiplicaciones matriciales.
- `tf.nn.relu` para las capas ocultas.
- `tf.sigmoid` para la salida binaria.

El entrenamiento se realiza con `tf.GradientTape`, que calcula los gradientes de la funcion de perdida respecto a los pesos. Luego se actualizan los parametros con una implementacion propia del optimizador Adam.

## Funciones de activacion

`ReLU` se usa en las capas ocultas porque permite aprender relaciones no lineales y reduce problemas de gradientes pequenos en comparacion con activaciones saturantes.

`Sigmoid` se usa en la salida porque produce un valor entre `0` y `1`, interpretable como probabilidad de bostezo.

## Epocas

Una epoca corresponde a una pasada completa del modelo sobre el conjunto de entrenamiento. Pocas epocas pueden causar subentrenamiento. Demasiadas epocas pueden causar sobreajuste. Por eso se aplica Early Stopping manual, deteniendo el entrenamiento cuando la perdida de validacion deja de mejorar.

## CUDA

TensorFlow puede usar GPU con CUDA cuando el entorno esta configurado correctamente. El codigo no fuerza el uso de GPU; TensorFlow la detecta automaticamente si esta disponible.
