# Analisis de metricas del modelo

## Archivos revisados

Las metricas del proyecto se encuentran en:

```text
metrics/results.csv
metrics/confusion_matrix.png
metrics/accuracy.png
metrics/loss.png
metrics/cross_validation_results.csv
metrics/cross_validation_summary.csv
metrics/test_error_analysis.csv
```

## Resultados principales

Segun `metrics/results.csv`, el modelo obtuvo:

| Metrica | Valor |
|---|---:|
| Accuracy | 0.8214 |
| Precision | 0.8462 |
| Recall | 0.7857 |
| F1-score | 0.8148 |

Segun `metrics/cross_validation_summary.csv`, la validacion cruzada reporta:

| Metrica | Media | Desviacion estandar |
|---|---:|---:|
| Accuracy | 0.8527 | 0.0853 |
| Precision | 0.8715 | 0.0766 |
| Recall | 0.8230 | 0.1099 |
| F1-score | 0.8461 | 0.0949 |

## Interpretacion

### Accuracy

El accuracy indica la proporcion total de predicciones correctas. Un valor de `0.8214` significa que el modelo acerto aproximadamente el 82.1% de las muestras de prueba.

En deteccion de bostezo, el accuracy es util, pero no debe evaluarse solo. Si el dataset esta desbalanceado, el accuracy puede ocultar errores importantes en una clase.

### Precision

La precision mide cuantas predicciones positivas realmente fueron positivas. En este proyecto, la clase positiva es `yawn`.

Una precision de `0.8462` indica que, cuando el modelo predice bostezo, suele acertar. Esto ayuda a reducir falsas alarmas.

### Recall

El recall mide cuantos bostezos reales fueron detectados. Un recall de `0.7857` indica que el modelo todavia puede dejar pasar algunos bostezos reales.

Para una aplicacion de somnolencia, el recall es importante porque un falso negativo puede significar no detectar un evento relevante.

### F1-score

El F1-score combina precision y recall. El valor `0.8148` muestra un desempeno balanceado, aunque todavia mejorable.

## Matriz de confusion

La matriz de confusion esta en:

```text
metrics/confusion_matrix.png
```

Permite observar:

- verdaderos positivos: bostezos detectados correctamente,
- verdaderos negativos: no bostezos detectados correctamente,
- falsos positivos: imagenes sin bostezo clasificadas como bostezo,
- falsos negativos: bostezos reales clasificados como no bostezo.

## Curvas de entrenamiento

Las graficas estan en:

```text
metrics/accuracy.png
metrics/loss.png
```

La curva de accuracy permite ver si el modelo mejora durante el entrenamiento. La curva de loss permite ver si la funcion de perdida disminuye.

Si el loss de entrenamiento baja mucho pero el de validacion empeora, puede existir sobreajuste. Si ambos se mantienen altos, puede faltar capacidad del modelo, datos o mejor preprocesamiento.

## Analisis de errores

El archivo:

```text
metrics/test_error_analysis.csv
```

sirve para revisar casos especificos donde el modelo falla. Este archivo es importante para identificar patrones como:

- boca parcialmente oculta,
- mala iluminacion,
- rostro fuera de centro,
- expresiones parecidas a bostezo,
- imagenes borrosas,
- recortes incorrectos de la region inferior del rostro.

## Limitaciones

- El dataset tiene 308 imagenes en total, pero no llega a 300 imagenes por clase.
- El rendimiento puede ser sensible al recorte de rostro/boca.
- La camara en vivo puede producir condiciones distintas a las imagenes de entrenamiento.
- El modelo MLP no aprovecha estructura espacial como lo haria una CNN.

## Conclusion

El modelo tiene resultados razonables para un proyecto academico, con precision superior a recall. Para una aplicacion real de alerta por somnolencia, convendria mejorar recall, aumentar el dataset y probar arquitecturas convolucionales o modelos especializados de vision.

