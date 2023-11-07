import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io

def predict_class(image):
    RGBImg = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    RGBImg= cv2.resize(RGBImg,(224,224))
    st.image(RGBImg, use_column_width=True)
    image = np.array(RGBImg) / 255.0
    new_model = tf.keras.models.load_model("Diabetic-Retinopat-CNN.model")
    predict=new_model.predict(np.array([image]))
    per=np.argmax(predict,axis=1)
    if per==1:
        return 'No DR'
    else:
        return 'DR'

st.title('Diabetic Retinopathy Detection')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    label = predict_class(image)
    st.write(label)