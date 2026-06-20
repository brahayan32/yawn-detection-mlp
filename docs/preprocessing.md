# Preprocesamiento

Cada imagen pasa por las siguientes etapas:

1. Lectura con OpenCV.
2. Conversion a escala de grises.
3. Suavizado Gaussiano con kernel `5x5`.
4. Redimensionamiento a `64x64` pixeles.
5. Normalizacion de pixeles dividiendo entre `255`.
6. Aplanado de la matriz `64x64` a un vector de `4096` valores.

## Pixeles

En imagenes de 8 bits, los pixeles tienen valores entre:

- `0`: negro
- `255`: blanco

Para entrenar redes neuronales es recomendable normalizar a un rango menor:

```text
pixel_normalizado = pixel_original / 255
```

Asi los valores quedan entre `0` y `1`, lo que mejora la estabilidad del entrenamiento.

## Filtros usados

La escala de grises reduce la cantidad de informacion y enfoca el modelo en formas e intensidad. El suavizado Gaussiano disminuye ruido visual. El redimensionamiento garantiza que todas las imagenes tengan la misma cantidad de caracteristicas.
