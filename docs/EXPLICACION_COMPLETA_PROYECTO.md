# Explicacion completa del proyecto: Deteccion de Bostezo con MLP

Este documento explica el funcionamiento completo del repositorio `yawn-detection-mlp` a partir de la evidencia encontrada en el codigo, la estructura de carpetas, los modelos exportados, las metricas y la aplicacion Streamlit.

> Nota: cuando un dato no aparece directamente en el codigo o en los archivos del repositorio, se indica explicitamente con la frase: "No se encontro evidencia directa en el codigo".

## 1. Resumen general del proyecto

El proyecto implementa un sistema de vision artificial para detectar bostezos en imagenes. Segun `README.md`, el sistema clasifica imagenes en dos clases:

```text
1 = yawn
0 = no_yawn
```

El problema se formula como **clasificacion binaria supervisada**: a partir de una imagen facial, el modelo estima una probabilidad de que la persona este bostezando. Si la probabilidad supera un umbral, se interpreta como `yawn`; si no lo supera, se interpreta como `no_yawn`.

Detectar bostezo significa identificar patrones visuales asociados a la apertura de la boca y a la region inferior del rostro. Por eso el preprocesamiento actual recorta la zona de boca/parte inferior del rostro antes de alimentar el modelo.

Tecnologias evidenciadas:

| Tecnologia | Evidencia | Uso |
|---|---|---|
| Python | `src/*.py`, `app_streamlit/*.py` | Lenguaje principal. |
| TensorFlow | `requirements.txt`, `src/model.py` | Modelo MLP, entrenamiento e inferencia. |
| OpenCV | `requirements.txt`, `src/preprocessing.py` | Lectura, conversion de color, deteccion facial, recorte, blur y redimensionamiento. |
| NumPy | `requirements.txt`, `src/model.py`, `src/preprocessing.py` | Vectores, matrices y datos numericos. |
| Pandas | `requirements.txt`, `src/evaluation.py`, `src/cross_validation.py` | Guardado de metricas CSV. |
| Matplotlib/Seaborn | `requirements.txt`, `src/evaluation.py` | Graficas y matriz de confusion. |
| Scikit-learn | `requirements.txt`, `src/evaluation.py`, `src/cross_validation.py` | Metricas y validacion cruzada. |
| Streamlit | `requirements.txt`, `app_streamlit/streamlit_app.py` | Interfaz grafica. |
| streamlit-webrtc | `requirements.txt`, `app_streamlit/streamlit_app.py` | Camara en vivo. |

Archivos principales que evidencian esto:

- `README.md`
- `requirements.txt`
- `src/config.py`
- `src/preprocessing.py`
- `src/model.py`
- `src/train.py`
- `app_streamlit/streamlit_app.py`
- `app_streamlit/utils.py`

## 2. Estructura general del repositorio

| Carpeta | Funcion | Se usa en la app? | Se usa en entrenamiento? | Observacion |
|---|---|---|---|---|
| `app_streamlit/` | Aplicacion visual en Streamlit. | Si | No directamente | Es la interfaz principal para usar el modelo. |
| `datasets/` | Imagenes organizadas por split y clase. | No | Si | Necesario para entrenar/evaluar. |
| `docs/` | Documentacion tecnica. | No | No | Sirve para exposicion y trazabilidad. |
| `metrics/` | Graficas y CSV de evaluacion. | Parcial | Si | La app busca `metrics/best_threshold.txt`, pero ese archivo no existe actualmente. |
| `models/` | Modelos TensorFlow SavedModel. | Si | Resultado del entrenamiento | La app carga `best_model` o `final_model`. |
| `notebooks/` | Analisis y experimentos. | No | Opcional | Sirven como evidencia del proceso. |
| `src/` | Codigo ML: config, preprocesamiento, modelo, entrenamiento, evaluacion. | Si | Si | La app depende de configuracion/preprocesamiento. |

Archivos importantes por carpeta:

- `app_streamlit/streamlit_app.py`: interfaz Streamlit.
- `app_streamlit/utils.py`: carga de modelo, umbral, preprocesamiento para app e inferencia.
- `src/config.py`: rutas, clases, hiperparametros y configuracion.
- `src/preprocessing.py`: pipeline de imagen.
- `src/model.py`: arquitectura MLP, perdida, optimizador y entrenamiento.
- `src/train.py`: flujo de entrenamiento y exportacion.
- `src/evaluation.py`: calculo y guardado de metricas.
- `models/best_model/`: primer modelo que intenta cargar la app.
- `models/final_model/`: modelo alternativo.

## 3. Creacion y organizacion de la base de datos / dataset

La carpeta `datasets/` esta organizada en tres particiones:

```text
datasets/train/
datasets/validation/
datasets/test/
```

Cada particion contiene dos clases:

```text
yawn
no_yawn
```

Rutas reales:

```text
datasets/train/yawn/
datasets/train/no_yawn/
datasets/validation/yawn/
datasets/validation/no_yawn/
datasets/test/yawn/
datasets/test/no_yawn/
```

Conteo actual de imagenes:

| Split | Clase | Imagenes |
|---|---:|---:|
| `train` | `no_yawn` | 88 |
| `train` | `yawn` | 114 |
| `validation` | `no_yawn` | 26 |
| `validation` | `yawn` | 27 |
| `test` | `no_yawn` | 27 |
| `test` | `yawn` | 26 |

Total observado: 308 imagenes.

La division en entrenamiento, validacion y prueba tiene esta finalidad:

- `train`: ajustar pesos del modelo.
- `validation`: evaluar durante entrenamiento y aplicar early stopping.
- `test`: evaluar rendimiento final en datos no usados durante entrenamiento.

Importancia de tener `yawn` y `no_yawn`:

- `yawn` representa la clase positiva: bostezo.
- `no_yawn` representa casos negativos: no hay bostezo.
- Sin ambas clases, el modelo no podria aprender a distinguir entre presencia y ausencia de bostezo.

Codigo que carga el dataset:

Archivo: `src/preprocessing.py`  
Funcion: `load_split`

```python
for class_name, label in CLASS_TO_LABEL.items():
    class_dir = split_dir / class_name
    if not class_dir.exists():
        continue
```

El codigo recorre carpetas cuyo nombre coincide con las clases definidas en `CLASS_TO_LABEL`.

Archivo: `src/config.py`

```python
CLASS_NAMES = ["no_yawn", "yawn"]
CLASS_TO_LABEL = {"no_yawn": 0, "yawn": 1}
LABEL_TO_CLASS = {0: "no_yawn", 1: "yawn"}
```

## 4. Preprocesamiento de imagenes

El preprocesamiento actual esta principalmente en:

- `src/preprocessing.py`
- `src/config.py`
- `app_streamlit/utils.py`

Configuracion base:

Archivo: `src/config.py`

```python
IMAGE_SIZE = (80, 80)
PREPROCESSING_STRATEGY = "lower_face"
```

Esto confirma que el proyecto usa imagenes redimensionadas a `80x80` y estrategia `lower_face`.

### 4.1 Conversion a escala de grises

La escala de grises representa cada pixel con un solo valor de intensidad, en vez de tres canales RGB. En este proyecto reduce la dimensionalidad y centra el analisis en formas, bordes e intensidad de la zona facial.

Archivo: `src/preprocessing.py`  
Funcion: `_load_preprocessed_crops`

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

Ventajas frente a RGB:

- Menos datos por imagen.
- Menor costo de computo.
- El MLP recibe un vector mas compacto.
- El color no parece ser la senal principal para detectar bostezo.

### 4.2 Recorte de la boca o parte inferior del rostro

El proyecto recorta una region centrada en la boca a partir del rostro detectado. Esto tiene sentido porque el bostezo se manifiesta principalmente en la apertura de la boca y la parte inferior del rostro.

Archivo: `src/preprocessing.py`  
Funcion: `crop_mouth_region`

```python
x1 = x + int(w * 0.10)
x2 = x + int(w * 0.90)
y1 = y + int(h * 0.42)
y2 = y + int(h * 1.04)
```

Ese fragmento usa las coordenadas del rostro detectado para seleccionar una zona horizontal amplia y verticalmente orientada hacia la parte inferior de la cara.

La funcion principal configurada es:

Archivo: `src/preprocessing.py`

```python
def crop_lower_face(gray_image: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for the mouth crop."""
    return crop_mouth_region(gray_image)
```

Si no se detecta rostro, el codigo usa un recorte alternativo:

Archivo: `src/preprocessing.py`  
Funcion: `_portrait_fallback_crop`

```python
x1 = int(width * 0.12)
x2 = int(width * 0.88)
y1 = int(height * 0.08)
y2 = int(height * 0.78)
```

### 4.3 Filtro GaussianBlur

`GaussianBlur` suaviza la imagen mediante un filtro gaussiano. Se usa para reducir ruido visual, pequeñas variaciones de textura e imperfecciones que podrian afectar el aprendizaje.

Archivo: `src/preprocessing.py`

```python
BLUR_KERNEL = (3, 3)
```

Archivo: `src/preprocessing.py`  
Funcion: `_vectorize_crop`

```python
blurred = cv2.GaussianBlur(gray_crop, BLUR_KERNEL, 0)
```

El kernel actual confirmado por el codigo es `3x3`.

### 4.4 Redimensionamiento

Todas las imagenes deben tener el mismo tamaño para que el MLP reciba siempre la misma cantidad de entradas. El proyecto usa `80x80`.

Archivo: `src/preprocessing.py`  
Funcion: `_vectorize_crop`

```python
resized = cv2.resize(blurred, image_size)
```

Archivo: `src/config.py`

```python
IMAGE_SIZE = (80, 80)
```

Como la imagen queda en escala de grises, `80 x 80 = 6400` caracteristicas.

### 4.5 Normalizacion

Normalizar pixeles significa llevar valores de `0-255` a `0-1`. Esto ayuda a estabilizar entrenamiento e inferencia porque evita entradas con escalas muy grandes.

Archivo: `src/preprocessing.py`  
Funcion: `_vectorize_crop`

```python
normalized = resized.astype("float32") / 255.0
```

### 4.6 Aplanado o flatten

El MLP trabaja con vectores, no con matrices de imagen. Por eso la imagen `80x80` se convierte en un vector de longitud `6400`.

Archivo: `src/preprocessing.py`  
Funcion: `_vectorize_crop`

```python
return normalized.flatten()
```

### 4.7 Forma final de entrada

En entrenamiento, `load_split` apila vectores en una matriz:

Archivo: `src/preprocessing.py`

```python
return np.vstack(features).astype("float32"), np.array(labels, dtype="int32"), paths
```

En la app Streamlit, la forma final se prepara asi:

Archivo: `app_streamlit/utils.py`  
Funcion: `preprocess_image_array`

```python
input_vector = normalized.flatten().reshape(1, -1)
```

Con `IMAGE_SIZE = (80, 80)`, esto produce una entrada de forma `(1, 6400)`.

## 5. Modelo de red neuronal MLP

Un MLP, o perceptron multicapa, es una red neuronal de capas densas. En este proyecto se usa para clasificacion binaria, ya que debe decidir entre `yawn` y `no_yawn`.

Archivo: `src/model.py`  
Clase: `PureTensorFlowMLP`

```python
def __init__(self, input_dim: int = 6400, seed: int = 42, name: str | None = None):
    super().__init__(name=name)
    tf.random.set_seed(seed)
```

Arquitectura real:

```python
self.w1 = tf.Variable(tf.random.normal([input_dim, 256], stddev=0.05), name="w1")
self.b1 = tf.Variable(tf.zeros([256]), name="b1")
self.w2 = tf.Variable(tf.random.normal([256, 64], stddev=0.05), name="w2")
self.b2 = tf.Variable(tf.zeros([64]), name="b2")
self.w3 = tf.Variable(tf.random.normal([64, 1], stddev=0.05), name="w3")
self.b3 = tf.Variable(tf.zeros([1]), name="b3")
```

Tabla de capas:

| Capa | Tamaño / neuronas | Activacion | Funcion |
|---|---:|---|---|
| Entrada | 6400 caracteristicas | No aplica | Recibe imagen preprocesada `80x80` aplanada. |
| Oculta 1 | 256 neuronas | ReLU | Aprende patrones no lineales. |
| Oculta 2 | 64 neuronas | ReLU | Combina representaciones intermedias. |
| Salida | 1 neurona | Sigmoid | Produce probabilidad de bostezo. |

Propagacion hacia adelante:

Archivo: `src/model.py`  
Funcion: `__call__`

```python
hidden_1 = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
hidden_2 = tf.nn.relu(tf.matmul(hidden_1, self.w2) + self.b2)
return tf.sigmoid(tf.matmul(hidden_2, self.w3) + self.b3)
```

La salida Sigmoid produce un valor entre `0` y `1`, interpretable como probabilidad de `yawn`.

## 6. Funciones de activacion

### 6.1 ReLU

ReLU significa Rectified Linear Unit. Devuelve `0` para valores negativos y mantiene valores positivos. Permite aprender relaciones no lineales y ayuda a evitar saturacion fuerte en capas ocultas.

Archivo: `src/model.py`

```python
hidden_1 = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
hidden_2 = tf.nn.relu(tf.matmul(hidden_1, self.w2) + self.b2)
```

En este proyecto se usa en las dos capas ocultas.

### 6.2 Sigmoid

Sigmoid transforma un valor real a un rango entre `0` y `1`. Es util para clasificacion binaria porque permite interpretar la salida como probabilidad.

Archivo: `src/model.py`

```python
return tf.sigmoid(tf.matmul(hidden_2, self.w3) + self.b3)
```

Si la salida es alta, el modelo considera mas probable que la imagen sea `yawn`.

## 7. Entrenamiento del modelo

El entrenamiento esta en:

- `src/train.py`
- `src/model.py`
- `src/evaluation.py`
- `notebooks/`

Configuracion de hiperparametros:

Archivo: `src/config.py`

```python
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 30
AUGMENT_TRAINING = True
L2_REGULARIZATION = 0.0001
```

Carga de datos:

Archivo: `src/train.py`

```python
data = load_all_dataset(DATASET_DIR)

x_train, y_train, _ = data["train"]
x_val, y_val, _ = data["validation"]
x_test, y_test, _ = data["test"]
```

Creacion del modelo:

Archivo: `src/train.py`

```python
model = PureTensorFlowMLP(input_dim=x_train.shape[1], seed=RANDOM_STATE)
```

Entrenamiento:

Archivo: `src/train.py`

```python
history = train_mlp(
    model,
    x_train,
    y_train,
    x_val=x_val if len(x_val) else None,
    y_val=y_val if len(y_val) else None,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    l2_strength=L2_REGULARIZATION,
)
```

Funcion de perdida:

Archivo: `src/model.py`  
Funcion: `binary_cross_entropy`

```python
y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
loss = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
return tf.reduce_mean(loss)
```

Optimizador Adam implementado manualmente:

Archivo: `src/model.py`  
Clase: `AdamOptimizer`

```python
self.m = [tf.Variable(tf.zeros_like(var), trainable=False) for var in variables]
self.v = [tf.Variable(tf.zeros_like(var), trainable=False) for var in variables]
```

Actualizacion de pesos:

Archivo: `src/model.py`

```python
variable.assign_sub(self.learning_rate * m_hat / (tf.sqrt(v_hat) + self.epsilon))
```

Ciclo de entrenamiento por lotes:

Archivo: `src/model.py`  
Funcion: `train_mlp`

```python
for start in range(0, len(indices), batch_size):
    batch_idx = indices[start : start + batch_size]
    x_batch = tf.convert_to_tensor(x_train[batch_idx], dtype=tf.float32)
    y_batch = tf.convert_to_tensor(y_train[batch_idx], dtype=tf.float32)
```

Early stopping:

Archivo: `src/model.py`

```python
if epochs_without_improvement >= patience:
    print("Entrenamiento detenido por Early Stopping.")
    break
```

Guardado del modelo:

Archivo: `src/train.py`

```python
save_model(model, MODELS_DIR / "best_model")
...
save_model(model, MODELS_DIR / "final_model")
```

Exportacion TensorFlow SavedModel:

Archivo: `src/model.py`

```python
def save_model(model, output_dir):
    tf.saved_model.save(model, str(output_dir))
```

Notebooks existentes:

- `notebooks/01_dataset_analysis.ipynb`
- `notebooks/02_preprocessing.ipynb`
- `notebooks/03_mlp_training.ipynb`
- `notebooks/04_cross_validation.ipynb`
- `notebooks/05_evaluation.ipynb`
- `notebooks/06_export_model.ipynb`

No se copian aqui porque son notebooks completos, pero sus rutas evidencian el flujo experimental.

## 8. Evaluacion del modelo

Archivo principal: `src/evaluation.py`

Metricas calculadas:

```python
return {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred, zero_division=0),
    "recall": recall_score(y_true, y_pred, zero_division=0),
    "f1_score": f1_score(y_true, y_pred, zero_division=0),
}
```

Significado:

- Accuracy: porcentaje total de aciertos.
- Precision: de los casos predichos como bostezo, cuantos realmente eran bostezo.
- Recall: de los bostezos reales, cuantos detecto el modelo.
- F1-score: balance entre precision y recall.
- Matriz de confusion: tabla de aciertos y errores por clase.

Matriz de confusion:

Archivo: `src/evaluation.py`

```python
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
```

Archivos actuales en `metrics/`:

| Archivo | Funcion |
|---|---|
| `metrics/accuracy.png` | Grafica de accuracy. |
| `metrics/loss.png` | Grafica de perdida. |
| `metrics/confusion_matrix.png` | Matriz de confusion. |
| `metrics/results.csv` | Metricas finales. |
| `metrics/cross_validation_results.csv` | Resultados por fold. |
| `metrics/cross_validation_summary.csv` | Media y desviacion de validacion cruzada. |
| `metrics/test_error_analysis.csv` | Analisis de errores en test. |

## 9. Umbral de decision

El umbral define a partir de que probabilidad se clasifica como bostezo.

En `src/predict_image.py`, el umbral fijo es:

```python
THRESHOLD = 0.5
```

La decision se hace asi:

```python
label = "yawn" if probability >= THRESHOLD else "no_yawn"
```

En la app Streamlit, el umbral se lee desde `metrics/best_threshold.txt` si existe:

Archivo: `app_streamlit/utils.py`  
Funcion: `read_default_threshold`

```python
threshold_path = METRICS_DIR / "best_threshold.txt"
if not threshold_path.exists():
    return 0.5
```

Actualmente, durante la inspeccion, `metrics/best_threshold.txt` no existe. Por tanto, la app usa `0.5`.

Sobre `src/tune_threshold.py`: no se encontro ese archivo en el repositorio actual.

## 10. Prediccion de una imagen

Prediccion por consola:

Archivo: `src/predict_image.py`

```python
model = tf.saved_model.load(args.model_dir)
x = preprocess_image(image_path).reshape(1, -1)
probability = float(model(tf.convert_to_tensor(x, dtype=tf.float32)).numpy().reshape(-1)[0])
```

Conversion a etiqueta:

```python
label = "yawn" if probability >= THRESHOLD else "no_yawn"
```

Prediccion en app Streamlit:

Archivo: `app_streamlit/utils.py`

```python
tensor = tf.convert_to_tensor(input_vector, dtype=tf.float32)
output = model(tensor).numpy().reshape(-1)
return float(output[0])
```

Clasificacion visual:

Archivo: `app_streamlit/utils.py`

```python
return "Bostezo detectado" if probability >= threshold else "No se detecta bostezo"
```

## 11. App Streamlit

La app esta en:

- `app_streamlit/streamlit_app.py`
- `app_streamlit/utils.py`
- `app_streamlit/README.md`

Ejecucion:

```powershell
py -m streamlit run app_streamlit/streamlit_app.py
```

Configuracion de pagina:

Archivo: `app_streamlit/streamlit_app.py`

```python
st.set_page_config(
    page_title="Detección de Bostezo",
    page_icon="camera",
    layout="centered",
)
```

Carga del modelo con cache:

```python
@st.cache_resource(show_spinner=False)
def get_cached_model():
    return load_trained_model()
```

Pestañas:

```python
upload_tab, camera_tab, live_tab = st.tabs(["Subir imagen", "Capturar foto", "Cámara en vivo"])
```

Subir imagen:

```python
uploaded_file = st.file_uploader(
    "Imagen",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    key=f"upload_{st.session_state.upload_widget_key}",
    label_visibility="collapsed",
)
```

Captura de foto:

```python
camera_file = st.camera_input(
    "Capturar imagen",
    key=f"camera_{st.session_state.camera_widget_key}",
    label_visibility="collapsed",
)
```

Estado de imagen:

```python
st.session_state.current_image_bytes = image_bytes
st.session_state.current_image_source = source
st.session_state.current_image_name = name
st.session_state.current_image_signature = signature
```

Boton de analizar:

```python
analyze = st.button("Analizar imagen", type="primary")
```

Inferencia al analizar:

```python
input_vector, _ = preprocess_image_array(image_bgr)
probability = predict_probability(model, input_vector)
st.session_state.probability = probability
st.session_state.result_label = classify_probability(probability, threshold)
```

Resultado:

```python
<div class="result-title">{label}</div>
<div class="probability">Probabilidad: {probability * 100:.1f}%</div>
```

Limpieza de estado:

```python
if st.button("Limpiar imagen"):
    clear_current_image()
    st.rerun()
```

Funcion de limpieza:

```python
st.session_state.current_image_bytes = None
st.session_state.result_label = None
st.session_state.probability = None
```

## 12. Camara en vivo

La implementacion con `streamlit-webrtc` existe.

Dependencia:

Archivo: `requirements.txt`

```text
streamlit-webrtc
```

Importacion:

Archivo: `app_streamlit/streamlit_app.py`

```python
import av
from streamlit_webrtc import webrtc_streamer
```

Si no esta instalada, la app muestra mensaje:

```python
if webrtc_streamer is None or av is None:
    st.error("La cámara en vivo requiere streamlit-webrtc...")
```

Procesador de video:

Archivo: `app_streamlit/streamlit_app.py`  
Clase: `YawnVideoProcessor`

```python
def recv(self, frame):
    image_bgr = frame.to_ndarray(format="bgr24")
    self.frame_count += 1
```

Prediccion cada 8 frames:

```python
if self.frame_count % self.frame_interval == 0:
    input_vector, _ = preprocess_image_array(image_bgr)
    probability = predict_probability(self.model, input_vector)
```

El valor `frame_interval` se configura aqui:

```python
frame_interval=8
```

El modelo no se recarga en cada frame porque se pasa al procesador desde el modelo ya cargado con `st.cache_resource`.

Creacion del streamer:

```python
ctx = webrtc_streamer(
    key="live-yawn-camera",
    video_processor_factory=lambda: YawnVideoProcessor(
        model=model,
        threshold=threshold,
        frame_interval=8,
    ),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
```

Limitaciones:

- Requiere permisos de camara del navegador.
- Puede funcionar mejor localmente que desplegado.
- Depende de `streamlit-webrtc` y `av`.
- Si la inferencia o el preprocesamiento son lentos, puede afectar fluidez.
- No se encontro evidencia directa en el codigo de pruebas automatizadas para WebRTC.

## 13. Relacion entre archivos

Flujo de usuario:

```text
Usuario
↓
Streamlit app
↓
Imagen subida / captura / camara en vivo
↓
Preprocesamiento
↓
Modelo MLP
↓
Probabilidad
↓
Resultado final
```

Flujo con rutas:

```text
app_streamlit/streamlit_app.py
↓ llama funciones de
app_streamlit/utils.py
↓ usa configuracion/preprocesamiento de
src/config.py y src/preprocessing.py
↓ carga modelo desde
models/best_model o models/final_model
↓ lee umbral desde
metrics/best_threshold.txt si existe; si no, usa 0.5
```

Para entrenamiento:

```text
src/train.py
↓ carga datos con
src/preprocessing.py
↓ crea modelo con
src/model.py
↓ evalua con
src/evaluation.py
↓ guarda modelos en
models/
↓ guarda metricas en
metrics/
```

## 14. Explicacion del codigo por partes

### Configuracion de tamaño de imagen

Archivo: `src/config.py`  
Funcion: configuracion global  
Codigo:

```python
IMAGE_SIZE = (80, 80)
PREPROCESSING_STRATEGY = "lower_face"
```

Explicacion: define el tamaño final de cada imagen y la estrategia de recorte.

### Recorte

Archivo: `src/preprocessing.py`  
Funcion: `crop_mouth_region`  
Codigo:

```python
y1 = y + int(h * 0.42)
y2 = y + int(h * 1.04)
```

Explicacion: toma una region inferior del rostro, enfocada en la boca.

### GaussianBlur

Archivo: `src/preprocessing.py`  
Funcion: `_vectorize_crop`  
Codigo:

```python
blurred = cv2.GaussianBlur(gray_crop, BLUR_KERNEL, 0)
```

Explicacion: suaviza la imagen antes de redimensionar.

### Normalizacion

Archivo: `src/preprocessing.py`  
Funcion: `_vectorize_crop`  
Codigo:

```python
normalized = resized.astype("float32") / 255.0
```

Explicacion: convierte pixeles de `0-255` a `0-1`.

### Flatten

Archivo: `src/preprocessing.py`  
Funcion: `_vectorize_crop`  
Codigo:

```python
return normalized.flatten()
```

Explicacion: convierte la imagen `80x80` en vector de `6400` valores.

### Arquitectura MLP

Archivo: `src/model.py`  
Funcion: `PureTensorFlowMLP.__init__`  
Codigo:

```python
self.w1 = tf.Variable(tf.random.normal([input_dim, 256], stddev=0.05), name="w1")
self.w2 = tf.Variable(tf.random.normal([256, 64], stddev=0.05), name="w2")
self.w3 = tf.Variable(tf.random.normal([64, 1], stddev=0.05), name="w3")
```

Explicacion: define dos capas ocultas y una salida binaria.

### Entrenamiento

Archivo: `src/model.py`  
Funcion: `train_mlp`  
Codigo:

```python
for epoch in range(1, epochs + 1):
    indices = np.random.permutation(len(x_train))
```

Explicacion: repite entrenamiento por epocas y mezcla datos antes de crear lotes.

### Funcion de perdida

Archivo: `src/model.py`  
Funcion: `binary_cross_entropy`  
Codigo:

```python
loss = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
```

Explicacion: mide error para clasificacion binaria.

### ReLU

Archivo: `src/model.py`  
Funcion: `__call__`  
Codigo:

```python
hidden_1 = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
```

Explicacion: activacion no lineal en capa oculta.

### Sigmoid

Archivo: `src/model.py`  
Funcion: `__call__`  
Codigo:

```python
return tf.sigmoid(tf.matmul(hidden_2, self.w3) + self.b3)
```

Explicacion: convierte salida en probabilidad.

### Prediccion

Archivo: `app_streamlit/utils.py`  
Funcion: `predict_probability`  
Codigo:

```python
output = model(tensor).numpy().reshape(-1)
return float(output[0])
```

Explicacion: ejecuta el modelo y devuelve probabilidad.

### Umbral

Archivo: `app_streamlit/utils.py`  
Funcion: `classify_probability`  
Codigo:

```python
return "Bostezo detectado" if probability >= threshold else "No se detecta bostezo"
```

Explicacion: convierte probabilidad en mensaje final.

### Carga del modelo

Archivo: `app_streamlit/utils.py`  
Funcion: `load_trained_model`  
Codigo:

```python
MODEL_CANDIDATES = (
    MODELS_DIR / "best_model",
    MODELS_DIR / "final_model",
)
```

Explicacion: intenta cargar primero `best_model` y luego `final_model`.

### Interfaz Streamlit

Archivo: `app_streamlit/streamlit_app.py`  
Funcion: `main`  
Codigo:

```python
upload_tab, camera_tab, live_tab = st.tabs(["Subir imagen", "Capturar foto", "Cámara en vivo"])
```

Explicacion: define los tres modos de entrada.

### Camara en vivo

Archivo: `app_streamlit/streamlit_app.py`  
Clase: `YawnVideoProcessor`  
Codigo:

```python
image_bgr = frame.to_ndarray(format="bgr24")
```

Explicacion: convierte cada frame de WebRTC a imagen compatible con OpenCV.

## 15. Limitaciones del proyecto

Limitaciones reales observadas:

1. **Dataset pequeño/moderado**: se observaron 308 imagenes. Para vision artificial generalizable, esto puede ser limitado.
2. **MLP no conserva estructura espacial como una CNN**: al usar `flatten`, la imagen se convierte en vector y se pierde organizacion espacial 2D.
3. **Sensibilidad al recorte**: si el detector facial no encuentra bien la cara, se usa fallback. Esto puede afectar prediccion.
4. **Camara en vivo depende de permisos**: WebRTC requiere permisos del navegador y puede comportarse distinto local/remoto.
5. **Posibles falsos positivos/falsos negativos**: toda clasificacion binaria puede equivocarse, especialmente con iluminacion, pose o boca parcialmente abierta.
6. **Umbral calibrado ausente**: `metrics/best_threshold.txt` no existe actualmente; la app usa `0.5`.
7. **No se encontro evidencia directa en el codigo** de pruebas automatizadas unitarias o de integracion.
8. **README menciona `src/webcam_demo.py`, pero ese archivo no existe actualmente** en la inspeccion realizada.

## 16. Conclusion

El proyecto logra integrar un flujo completo de deteccion de bostezo:

1. Dataset organizado por clases y particiones.
2. Preprocesamiento con escala de grises, recorte de boca/rostro inferior, blur, resize, normalizacion y flatten.
3. Modelo MLP implementado con TensorFlow puro.
4. Entrenamiento con binary cross entropy, Adam manual, validacion y early stopping.
5. Evaluacion con accuracy, precision, recall, F1-score y matriz de confusion.
6. Exportacion de modelos TensorFlow SavedModel.
7. App Streamlit para usar imagen subida, captura de foto y camara en vivo.

Streamlit aporta una forma sencilla de exponer el modelo sin construir un frontend separado. `streamlit-webrtc` permite una experiencia de camara en vivo, aunque depende de permisos del navegador y puede requerir ajustes en despliegue.

Mejoras futuras razonables:

- Aumentar y balancear dataset.
- Evaluar una CNN para conservar estructura espacial.
- Agregar pruebas automatizadas para preprocesamiento e inferencia.
- Documentar mejor el origen del dataset.
- Agregar o regenerar `metrics/best_threshold.txt` si se quiere usar un umbral calibrado.
- Corregir referencias obsoletas del README si el archivo `src/webcam_demo.py` ya no existe.

