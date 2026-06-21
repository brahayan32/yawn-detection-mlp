# Preprocesamiento

Cada imagen pasa por las siguientes etapas:

1. Lectura con OpenCV.
2. Conversion a escala de grises.
3. Deteccion de rostro con OpenCV.
4. Recorte centrado en la boca a partir del rostro detectado.
5. Suavizado Gaussiano ligero con kernel `3x3`.
6. Redimensionamiento a `80x80` pixeles.
7. Normalizacion de pixeles dividiendo entre `255`.
8. Aplanado de la matriz `80x80` a un vector de `6400` valores.

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

Durante el entrenamiento se aplican aumentaciones ligeras en memoria: pequenas rotaciones, desplazamientos y cambios de brillo/contraste. Estas variantes no modifican el dataset original y no se aplican en validacion ni prueba.

## Recorte de boca

La estrategia configurada es `lower_face`, pero actualmente aplica un recorte centrado en la region de la boca. Primero se detecta el rostro con un clasificador Haar de OpenCV. Para mejorar la deteccion, la imagen se ecualiza y se prueban pequenas rotaciones cuando la cabeza esta inclinada.

Luego se toma una zona central e inferior del rostro, incluyendo boca completa y menton. Esto reduce informacion irrelevante como ojos, cabello, fondo, ropa o iluminacion. Si no se detecta rostro, se usa un recorte de respaldo tipo retrato, mas enfocado en la zona central-superior de la imagen.
