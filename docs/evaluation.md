# Evaluacion

El modelo se evalua con el conjunto de prueba, que no debe usarse durante entrenamiento ni validacion.

## Metricas

- Accuracy: porcentaje total de predicciones correctas.
- Precision: proporcion de predicciones positivas que realmente eran bostezos.
- Recall: proporcion de bostezos reales que el modelo detecto.
- F1 score: promedio armonico entre precision y recall.
- Matriz de confusion: tabla de aciertos y errores por clase.

## Validacion cruzada

La validacion cruzada divide el conjunto de entrenamiento en varios folds. En cada iteracion se entrena con una parte y se valida con otra. Esto permite estimar mejor la capacidad de generalizacion del modelo y reducir la dependencia de una sola division de datos.

El proyecto guarda:

- `metrics/cross_validation_results.csv`
- `metrics/cross_validation_summary.csv`
