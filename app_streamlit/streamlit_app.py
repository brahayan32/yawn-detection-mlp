from __future__ import annotations

import hashlib
import sys
import threading
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_streamlit.utils import (  # noqa: E402
    bgr_to_rgb,
    check_runtime_dependencies,
    classify_probability,
    decode_image_bytes,
    load_trained_model,
    predict_probability,
    preprocess_image_array,
    read_default_threshold,
)

try:
    import av
    from streamlit_webrtc import webrtc_streamer
except ImportError:  # pragma: no cover - handled in the UI.
    av = None
    webrtc_streamer = None


st.set_page_config(
    page_title="Detección de Bostezo",
    page_icon="camera",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def get_cached_model():
    return load_trained_model()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #0f172a;
            color: #f8fafc;
        }
        .main .block-container {
            max-width: 820px;
            padding-top: 2.2rem;
            padding-bottom: 2.5rem;
        }
        h1, h2, h3, p, span, label {
            color: #f8fafc;
            letter-spacing: 0;
        }
        .app-header {
            text-align: center;
            margin-bottom: 1.35rem;
        }
        .app-header h1 {
            margin-bottom: 0.45rem;
            font-size: 2.25rem;
            font-weight: 850;
            color: #f8fafc;
        }
        .app-header p {
            margin: 0 auto;
            max-width: 600px;
            color: #cbd5e1;
            font-size: 1rem;
            line-height: 1.55;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #111827;
            border: 1px solid #334155;
            border-radius: 20px;
            box-shadow: 0 18px 48px rgba(0, 0, 0, 0.32);
        }
        .section-title {
            color: #f8fafc;
            font-size: 1.03rem;
            font-weight: 800;
            margin: 0 0 0.75rem 0;
        }
        .hint {
            color: #cbd5e1;
            font-size: 0.95rem;
            margin: 0.15rem 0 0.85rem 0;
        }
        .current-source {
            color: #7dd3fc;
            font-size: 0.92rem;
            font-weight: 700;
            margin: 0.4rem 0 0.85rem 0;
        }
        .result-card {
            border-radius: 18px;
            padding: 1.3rem;
            margin-top: 1rem;
            text-align: center;
        }
        .result-waiting {
            background: #1e293b;
            border: 1px solid #38bdf8;
        }
        .result-yawn {
            background: #7f1d1d;
            border: 1px solid #fb923c;
        }
        .result-clear {
            background: #14532d;
            border: 1px solid #22c55e;
        }
        .result-title {
            color: #fff7ed;
            font-size: 1.85rem;
            line-height: 1.15;
            font-weight: 850;
            margin-bottom: 0.55rem;
        }
        .probability {
            color: #f8fafc;
            font-size: 1.2rem;
            font-weight: 750;
        }
        .live-note {
            color: #cbd5e1;
            font-size: 0.92rem;
            margin-top: 0.75rem;
        }
        .stButton > button {
            width: 100%;
            border-radius: 13px;
            border: 1px solid #38bdf8;
            background: #0284c7;
            color: #ffffff;
            font-weight: 800;
            padding: 0.78rem 1rem;
        }
        .stButton > button:hover {
            border-color: #7dd3fc;
            background: #0369a1;
            color: #ffffff;
        }
        div[data-testid="stAlert"] {
            background: #1e293b;
            color: #f8fafc;
            border-color: #334155;
        }
        div[data-testid="stAlert"] p {
            color: #f8fafc;
        }
        button[data-baseweb="tab"] p {
            color: #cbd5e1;
            font-weight: 700;
        }
        button[data-baseweb="tab"][aria-selected="true"] p {
            color: #38bdf8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    defaults = {
        "current_image_bytes": None,
        "current_image_source": None,
        "current_image_name": None,
        "current_image_signature": None,
        "result_label": None,
        "probability": None,
        "upload_widget_key": 0,
        "camera_widget_key": 0,
        "last_upload_signature": None,
        "last_camera_signature": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def image_signature(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def clear_result() -> None:
    st.session_state.result_label = None
    st.session_state.probability = None


def set_current_image(image_bytes: bytes, source: str, name: str) -> None:
    signature = image_signature(image_bytes)
    if (
        signature != st.session_state.current_image_signature
        or source != st.session_state.current_image_source
    ):
        st.session_state.current_image_bytes = image_bytes
        st.session_state.current_image_source = source
        st.session_state.current_image_name = name
        st.session_state.current_image_signature = signature
        clear_result()


def clear_current_image() -> None:
    st.session_state.current_image_bytes = None
    st.session_state.current_image_source = None
    st.session_state.current_image_name = None
    st.session_state.current_image_signature = None
    st.session_state.last_upload_signature = None
    st.session_state.last_camera_signature = None
    st.session_state.upload_widget_key += 1
    st.session_state.camera_widget_key += 1
    clear_result()


def source_label(source: str | None) -> str:
    if source == "upload":
        return "Imagen subida"
    if source == "camera":
        return "Captura de cámara"
    return "Sin imagen"


def show_result() -> None:
    probability = st.session_state.probability
    label = st.session_state.result_label
    if probability is None or label is None:
        return

    result_class = "result-yawn" if label == "Bostezo detectado" else "result-clear"
    st.markdown(
        f"""
        <div class="result-card {result_class}">
            <div class="result-title">{label}</div>
            <div class="probability">Probabilidad: {probability * 100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_live_result(label: str | None, probability: float | None) -> None:
    if label is None or probability is None:
        st.markdown(
            """
            <div class="result-card result-waiting">
                <div class="result-title">Esperando cámara...</div>
                <div class="probability">Probabilidad: --</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    result_class = "result-yawn" if label == "Bostezo detectado" else "result-clear"
    st.markdown(
        f"""
        <div class="result-card {result_class}">
            <div class="result-title">{label}</div>
            <div class="probability">Probabilidad: {probability * 100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


class YawnVideoProcessor:
    def __init__(self, model, threshold: float, frame_interval: int = 8):
        self.model = model
        self.threshold = threshold
        self.frame_interval = frame_interval
        self.frame_count = 0
        self.result_label = None
        self.probability = None
        self.error = None
        self.lock = threading.Lock()

    def recv(self, frame):
        image_bgr = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % self.frame_interval == 0:
            try:
                input_vector, _ = preprocess_image_array(image_bgr)
                probability = predict_probability(self.model, input_vector)
                label = classify_probability(probability, self.threshold)
                with self.lock:
                    self.probability = probability
                    self.result_label = label
                    self.error = None
            except Exception as exc:  # pragma: no cover - depends on webcam/runtime.
                with self.lock:
                    self.error = str(exc)

        with self.lock:
            label = self.result_label
            probability = self.probability

        if label is not None and probability is not None:
            color = (0, 140, 255) if label == "Bostezo detectado" else (60, 190, 95)
            overlay = f"{label} ({probability * 100:.1f}%)"
            cv2 = __import__("cv2")
            cv2.putText(
                image_bgr,
                overlay,
                (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                color,
                2,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(image_bgr, format="bgr24")

    def get_result(self) -> tuple[str | None, float | None, str | None]:
        with self.lock:
            return self.result_label, self.probability, self.error


def main() -> None:
    init_state()
    inject_styles()

    st.markdown(
        """
        <div class="app-header">
            <h1>Detección de Bostezo</h1>
            <p>Sube una imagen o usa la cámara para analizar si hay bostezo.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    missing_dependencies = check_runtime_dependencies()
    threshold = read_default_threshold()
    model = None
    model_error = None

    if not missing_dependencies:
        try:
            model, _ = get_cached_model()
        except Exception as exc:
            model_error = exc

    with st.container(border=True):
        st.markdown('<p class="section-title">Selecciona una imagen</p>', unsafe_allow_html=True)

        upload_tab, camera_tab, live_tab = st.tabs(["Subir imagen", "Capturar foto", "Cámara en vivo"])

        with upload_tab:
            st.markdown(
                '<p class="hint">Elige una imagen facial desde tu computador.</p>',
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "Imagen",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key=f"upload_{st.session_state.upload_widget_key}",
                label_visibility="collapsed",
            )

        with camera_tab:
            st.markdown(
                '<p class="hint">Captura una imagen desde la cámara del navegador.</p>',
                unsafe_allow_html=True,
            )
            camera_file = st.camera_input(
                "Capturar imagen",
                key=f"camera_{st.session_state.camera_widget_key}",
                label_visibility="collapsed",
            )

        with live_tab:
            st.markdown(
                '<p class="hint">Activa la cámara para ver el resultado actualizado en vivo.</p>',
                unsafe_allow_html=True,
            )

            if webrtc_streamer is None or av is None:
                st.error(
                    "La cámara en vivo requiere streamlit-webrtc. Instala dependencias con: "
                    "py -m pip install -r requirements.txt"
                )
            elif missing_dependencies:
                st.error("Faltan dependencias necesarias: " + ", ".join(missing_dependencies))
            elif model_error is not None:
                st.error(f"No se pudo cargar el modelo entrenado: {model_error}")
            elif model is None:
                st.error("No se encontró un modelo disponible para cámara en vivo.")
            else:
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

                result_placeholder = st.empty()
                st.markdown(
                    '<p class="live-note">La inferencia se ejecuta cada 8 frames para mantener la app fluida.</p>',
                    unsafe_allow_html=True,
                )

                if ctx.video_processor:
                    while ctx.state.playing:
                        label, probability, error = ctx.video_processor.get_result()
                        with result_placeholder.container():
                            if error:
                                st.warning("No se pudo procesar un frame. La cámara seguirá intentando.")
                            show_live_result(label, probability)
                        time.sleep(0.15)
                else:
                    with result_placeholder.container():
                        show_live_result(None, None)

        upload_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
        camera_bytes = camera_file.getvalue() if camera_file is not None else None
        upload_signature = image_signature(upload_bytes) if upload_bytes is not None else None
        camera_signature = image_signature(camera_bytes) if camera_bytes is not None else None

        upload_changed = (
            upload_signature is not None
            and upload_signature != st.session_state.last_upload_signature
        )
        camera_changed = (
            camera_signature is not None
            and camera_signature != st.session_state.last_camera_signature
        )

        if upload_changed and upload_bytes is not None:
            st.session_state.last_upload_signature = upload_signature
            st.session_state.last_camera_signature = camera_signature
            st.session_state.camera_widget_key += 1
            set_current_image(upload_bytes, "upload", uploaded_file.name)
        elif camera_changed and camera_bytes is not None:
            st.session_state.last_camera_signature = camera_signature
            st.session_state.last_upload_signature = upload_signature
            st.session_state.upload_widget_key += 1
            set_current_image(camera_bytes, "camera", "captura_camara.jpg")

        st.markdown('<p class="section-title">Vista previa de la imagen actual</p>', unsafe_allow_html=True)
        if st.session_state.current_image_bytes is None:
            st.info("Sube una imagen o captura una foto para comenzar.")
        else:
            try:
                preview_bgr = decode_image_bytes(st.session_state.current_image_bytes)
                st.markdown(
                    f'<p class="current-source">{source_label(st.session_state.current_image_source)}</p>',
                    unsafe_allow_html=True,
                )
                st.image(bgr_to_rgb(preview_bgr), width="stretch")
            except Exception:
                st.error("No se pudo procesar la imagen actual. Inténtalo de nuevo.")
                clear_current_image()

        analyze = st.button("Analizar imagen", type="primary")

        if analyze:
            if missing_dependencies:
                st.error("Faltan dependencias necesarias: " + ", ".join(missing_dependencies))
                return

            if model_error is not None:
                st.error(f"No se pudo cargar el modelo entrenado: {model_error}")
                return

            if model is None:
                st.error("No se encontró un modelo disponible para analizar la imagen.")
                return

            if st.session_state.current_image_bytes is None:
                st.warning("Primero sube una imagen o captura una foto con la cámara.")
                return

            try:
                image_bgr = decode_image_bytes(st.session_state.current_image_bytes)
                input_vector, _ = preprocess_image_array(image_bgr)
                probability = predict_probability(model, input_vector)
                st.session_state.probability = probability
                st.session_state.result_label = classify_probability(probability, threshold)
            except Exception:
                clear_result()
                st.error("No se pudo procesar la imagen, inténtalo de nuevo.")

        show_result()

        if st.button("Limpiar imagen"):
            clear_current_image()
            st.rerun()


if __name__ == "__main__":
    main()
