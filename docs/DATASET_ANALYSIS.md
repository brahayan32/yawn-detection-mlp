# Analisis del dataset

## Objetivo

Este documento resume el estado real del dataset usado en el proyecto `yawn-detection-mlp`. Los conteos se obtienen con:

```powershell
py tools/dataset_summary.py
```

El script cuenta archivos de imagen en:

```text
datasets/train/no_yawn
datasets/train/yawn
datasets/validation/no_yawn
datasets/validation/yawn
datasets/test/no_yawn
datasets/test/yawn
```

## Resumen actual

Conteo observado en el repositorio:

| Split | no_yawn | yawn | Total |
|---|---:|---:|---:|
| train | 88 | 114 | 202 |
| validation | 26 | 27 | 53 |
| test | 27 | 26 | 53 |
| **Total** | **141** | **167** | **308** |

## Balance entre clases

El dataset tiene:

- `no_yawn`: 141 imagenes.
- `yawn`: 167 imagenes.
- Total: 308 imagenes.

La clase `yawn` tiene 26 imagenes mas que `no_yawn`. El balance es aceptable para una prueba academica inicial, pero no es ideal para un sistema robusto.

## Cumplimiento del minimo de 300 imagenes por clase

La rubrica solicita revisar si se cumple el minimo de 300 imagenes por clase.

Estado actual:

| Clase | Imagenes actuales | Minimo esperado | Cumple |
|---|---:|---:|---|
| no_yawn | 141 | 300 | No |
| yawn | 167 | 300 | No |

El proyecto no cumple el minimo de 300 imagenes por clase. Esta limitacion debe mencionarse en el reporte y en la exposicion.

## Origen y variedad

En los archivos actuales del repositorio no se observa una ficha formal del origen del dataset. Por eso no se debe afirmar un origen especifico si no esta documentado por el equipo.

Aspectos que deben revisarse manualmente si se quiere fortalecer el reporte:

- variedad de personas,
- iluminacion,
- angulos de camara,
- distancia al rostro,
- expresiones similares a bostezo,
- calidad y resolucion de las imagenes.

## Limitaciones reales

- Menos de 300 imagenes por clase.
- Dataset pequeno para generalizar a diferentes personas y condiciones de luz.
- Balance moderado, pero no perfecto.
- No hay origen formal documentado en los archivos revisados.
- El rendimiento del modelo puede cambiar con imagenes fuera del entorno del dataset.

## Evidencias generadas

El script genera:

```text
metrics/dataset_summary.csv
metrics/dataset_distribution.png
```

Estas evidencias deben incluirse en el reporte final de la rubrica.
