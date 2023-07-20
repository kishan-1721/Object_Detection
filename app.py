import streamlit as st
from main import main
from ultralytics import YOLO
import shutil

model = YOLO('yolov8m.pt')

st.title('Object Detection')

main()

if st.button("Detect Object"):
    model.predict(source = 'processed_image.jpg', save = True, conf = 0.7)
    st.image('runs/detect/predict/processed_image.jpg')
    shutil.rmtree('runs')

# shutil.rmtree('runs')