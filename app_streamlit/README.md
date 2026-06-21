# Aplicacion Streamlit para deteccion de bostezo

Esta carpeta contiene una app sencilla en Streamlit para usar el modelo entrenado del proyecto `yawn-detection-mlp`.

## Que hace

La app permite usar tres modos:

- `Subir imagen`: cargar una imagen desde el computador.
- `Capturar foto`: tomar una foto con `st.camera_input`.
- `Camara en vivo`: usar video en vivo con `streamlit-webrtc`.

En todos los modos se usa el modelo MLP existente para mostrar:

- `Bostezo detectado`
- `No se detecta bostezo`
- probabilidad estimada

La app no entrena modelos nuevos y no modifica datasets, modelos ni metricas.

## Como ejecutar

Desde la raiz del proyecto:

```powershell
py -m streamlit run app_streamlit/streamlit_app.py
```

Si faltan dependencias:

```powershell
py -m pip install -r requirements.txt
```

## Como probar

1. Ejecuta la app.
2. Elige `Subir imagen`, `Capturar foto` o `Camara en vivo`.
3. En imagen/foto, presiona `Analizar imagen`.
4. En camara en vivo, presiona `START` y permite el acceso a la camara del navegador.
5. Revisa el resultado y la probabilidad.

## Camara en vivo

La camara en vivo usa `streamlit-webrtc`. Esta opcion procesa frames del navegador y ejecuta inferencia cada cierto numero de frames para evitar que la app se vuelva lenta.

En local suele funcionar mejor. El navegador puede pedir permisos de camara y, en despliegues remotos, puede requerir configuracion adicional de HTTPS/permisos.

## Archivos que usa internamente

- `models/best_model`
- `models/final_model`
- `metrics/best_threshold.txt`

## Preprocesamiento

La app usa el mismo enfoque del proyecto actual:

- escala de grises,
- recorte configurado en el proyecto,
- suavizado Gaussiano configurado,
- redimensionamiento a `80x80`,
- normalizacion,
- vector final de `6400` caracteristicas.

