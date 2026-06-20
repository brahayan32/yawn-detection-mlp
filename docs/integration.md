# Integracion

El modelo final se exporta con TensorFlow SavedModel, sin Keras:

```text
models/final_model/
```

Para integrarlo en backend, se debe aplicar exactamente el mismo preprocesamiento usado durante entrenamiento:

1. Convertir frame o imagen a escala de grises.
2. Aplicar suavizado Gaussiano.
3. Redimensionar a `64x64`.
4. Normalizar dividiendo entre `255`.
5. Aplanar a vector de `4096` caracteristicas.
6. Enviar al modelo.

## Interpretacion de salida

```text
salida >= 0.5 -> yawn
salida < 0.5  -> no_yawn
```

El umbral `0.5` puede ajustarse si las metricas muestran muchos falsos positivos o falsos negativos.

## Carga del modelo

```python
import tensorflow as tf

model = tf.saved_model.load("models/final_model")
probability = model(input_vector)
```
