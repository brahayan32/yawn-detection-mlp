# Yawn Detection MLP

Sistema de vision artificial para clasificar imagenes en dos clases:

- `1` = yawn
- `0` = no_yawn

El modelo principal es un MLP implementado con TensorFlow puro, sin Keras. Las imagenes se preprocesan en escala de grises, recorte de boca, suavizado Gaussiano ligero, redimensionamiento a `80x80`, normalizacion de pixeles de `0-255` a `0-1` y vectorizacion a `6400` caracteristicas.

## Estructura

```text
yawn-detection-mlp/
|-- datasets/
|-- notebooks/
|-- src/
|-- models/
|-- metrics/
|-- docs/
|-- requirements.txt
|-- environment.yml
`-- README.md
```

## Instalacion

Con Miniconda:

```bash
conda env create -f environment.yml
conda activate yawn-detection-mlp
jupyter lab
```

Tambien se puede instalar con `pip`:

```bash
pip install -r requirements.txt
```

## Dataset esperado

```text
datasets/
|-- train/
|   |-- yawn/
|   `-- no_yawn/
|-- validation/
|   |-- yawn/
|   `-- no_yawn/
`-- test/
    |-- yawn/
    `-- no_yawn/
```

## Entrenamiento

Desde la raiz del proyecto:

```bash
python -m src.train
```

## Validacion cruzada

```bash
python -m src.cross_validation
```

## Probar una imagen nueva

```bash
python -m src.predict_image ruta/a/tu_imagen.jpg
```

La salida muestra la probabilidad de `yawn` y la clase final.

## Probar con camara

```bash
python -m src.webcam_demo
```

Presiona `q` para cerrar la ventana.

## Salidas

- Modelo final exportado con TensorFlow SavedModel: `models/final_model/`
- Mejor modelo exportado con TensorFlow SavedModel: `models/best_model/`
- Graficas: `metrics/accuracy.png`, `metrics/loss.png`
- Matriz de confusion: `metrics/confusion_matrix.png`
- Resultados: `metrics/results.csv`
