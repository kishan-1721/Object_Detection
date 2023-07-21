import streamlit as st
from PIL import Image
import cv2
import numpy as np

def main():

    # st.title("Image Processing and Saving Example")

    # Upload an image using Streamlit's file uploader

    genre = st.radio(
        "How You Want To Upload Your Image",
        ('Browse Photos', 'Camera'))

    if genre == 'Camera':
        uploaded_image = st.camera_input("Take a picture")
    else:
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    # uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        # st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Convert the image to a format compatible with PIL and OpenCV
        pil_image = Image.open(uploaded_image)
        opencv_image = np.array(pil_image)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

        # Image processing code (You can add any processing you want here)

        # Save the processed image using PIL
        # st.write("Processed Image")
        # st.image(pil_image, caption="Processed Image", use_column_width=True)

        # Save the processed image using OpenCV
        # save_button = st.button("Save Processed Image")
        # if save_button:
            # Provide a file path to save the image
        save_path = "processed_image.jpg"  # You can change the file format or filename here
        cv2.imwrite(save_path, opencv_image)
        st.success(f"Image saved as {save_path}")

# if __name__ == "__main__":
#     main()
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