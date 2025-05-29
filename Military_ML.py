import streamlit as st
from PIL import Image
import torch
import sys
import numpy as np
import pandas as pd
from io import BytesIO
import platform
import pathlib

# For Fix PosixPath issue on Windows
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

#  Yolo Path
YOLOV5_PATH = r"C:\Users\achu1\Documents\GUVI\Streamlit\yolov5"
sys.path.append(YOLOV5_PATH)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator
from utils.augmentations import letterbox
from utils.torch_utils import select_device

# Threat classification 
threat_map = {
    "Marksman": "Threat",
    "Weapon": "Threat",
    "Tank": "Threat",
    "War Truck": "Threat",
    "Missile": "Threat",
    "War Plane": "Threat",
    "Soldier": "Non-Threat",
    "Jeep": "Non-Threat",
    "Ground": "Non-Threat",
    "Helicopter": "Non-Threat",
    "Radar": "Non-Threat"
}

def classify_threat(label):
    return threat_map.get(label, "Unknown")

# Loading YOLOv5 model 
@st.cache_resource
def load_model():
    device = select_device("cpu")  
    model_path = r"C:\Users\achu1\Documents\GUVI\Project-6\best.pt"
    model = DetectMultiBackend(model_path, device=device)
    model.eval()
    return model, device

model, device = load_model()


st.title("Military Object Detection & Threat Classification")
st.write("Upload an image to detect objects and classify them as **Threat** or **Non-Threat**.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    img = np.array(image)
    img_resized = letterbox(img, new_shape=640)[0]
    img_resized = img_resized.transpose((2, 0, 1))  # HWC to CHW
    img_resized = np.ascontiguousarray(img_resized)
    img_tensor = torch.from_numpy(img_resized).to(device).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    pred = model(img_tensor)
    pred = non_max_suppression(pred)[0]

    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], img.shape).round()

        annotator = Annotator(img.copy(), line_width=2, example=str(model.names))
        results = []

        for *xyxy, conf, cls in pred:
            label = model.names[int(cls)]
            confidence = float(conf)
            threat = classify_threat(label)

            results.append({
                "Object": label,
                "Threat Level": threat,
                "Confidence": f"{confidence:.2f}"
            })

            color = (255, 0, 0) if threat == "Threat" else (0, 255, 0)
            box_label = f"{label} ({threat}) {confidence:.2f}"
            annotator.box_label(xyxy, box_label, color=color)

        st.subheader("Detection Results")
        st.dataframe(pd.DataFrame(results))


        st.subheader("🖼️ Annotated Image")
        annotated_img = annotator.result()
        st.image(annotated_img, use_container_width=True)



    else:
        st.warning(" No objects detected.")

     