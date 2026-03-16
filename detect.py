import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("YOLO Object Detection")

st.write("Lade ein Bild hoch und YOLO erkennt die Objekte.")

# Modell laden
model = YOLO("yolov8n.pt")

# Bild hochladen
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    img_array = np.array(image)

    # Objekte erkennen
    results = model(img_array)

    # Ergebnisbild erzeugen
    result_image = results[0].plot()

    st.image(result_image, caption="Erkannte Objekte", use_column_width=True)
