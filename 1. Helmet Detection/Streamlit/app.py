import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import av

# Load YOLO model
model = YOLO("1. Helmet Detection/Models/best.pt")
model.model.names = {0: "Helmet", 1: "No Helmet", 2: "No Person"}

st.set_page_config(page_title="Helmet Detection", layout="centered")
st.title("Real-time Helmet Detection")

class HelmetDetector(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO inference
        results = self.model(img, verbose=False)
        r = results[0]
        annotated = r.plot()

        # Optionally, overlay count text
        import cv2
        cv2.putText(
            annotated,
            f"Detections: {len(r.boxes)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            1,
        )

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

webrtc_streamer(
    key="helmet-detection",
    video_transformer_factory=HelmetDetector,
    media_stream_constraints={"video": True, "audio": False},
)