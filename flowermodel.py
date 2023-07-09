import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageOps


st.title("Farmer Flower Image Recognition App")
st.image("farm.jpg")
st.write("This is an App intended to help farmers identify flowers in their farmlands")

st.title("Upload the Flower Image")
st.write("""Please upload your choice of flower to predict""")
uploaded_file = st.file_uploader("Choose a jpeg file", type=["jfif", "jpg", "jpeg"])
new_model = load_model("flowerprediction (2).h5")
if uploaded_file is None:
    st.write("Please upload a jpg, jpeg or a jfif image of the flower in the drag and drop box above")
else:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    size = (256, 256)
    method = Image.NEAREST if image.size == size else Image.ANTIALIAS
    image = ImageOps.fit(image, size, method=method, centering=(0.5, 0.5))
    image = np.asarray(image)
    st.header("Detect the FLower Type")
    st.write("Now click the button below to identify the FLower")
    if st.button('Make Prediction'):
        class_names = np.array(['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'])
        predictions = new_model.predict(np.expand_dims(image / 255, 0))
        score = tf.nn.softmax(predictions)
        prediction = np.argmax(score, axis=1)
        confidence = round(np.max(predictions) * 100, 2)
        if prediction == 0 and confidence > 70:
            st.write(f"This is {class_names[0].upper()} and I am  {confidence}% about it")
            st.header("Importance of  Daisy Flowers to Agriculture")
            st.image("DAISY.jfif")
            st.markdown(
                """
                The following list won't indent no matter what I try:
                - Item 1
                - Item 2
                - Item 3
                """
            )
            st.write("")
        elif prediction == 1 and confidence > 70:
            st.write(f"This is {class_names[1].upper()} and I am  {confidence}% about it")
            st.header("Importance of Dandelion Flowers to Agriculture")
            st.image("Dandelions.jfif")
            st.write("Dandelions flowers are blablabla")
        elif prediction == 2 and confidence > 70:
            st.write(f"This is {class_names[2].upper()} and I am  {confidence}% about it")
            st.header("Importance of Roses Flowers to Agriculture")
            st.image("ROSES.jfif")
            st.markdown(
                """
                These are some of the reasons why you should leave this flower on your field:
                - Roses are efficient pollinators due to their attractive flowers and abundant pollen production.
                - Their abundant pollen increases the chances of successful pollination for agricultural plants.
                - Roses attract bees, crucial pollinators for crops, promoting their activity and improving yields.
                - Incorporating roses enhances biodiversity by providing food and habitat for insects and birds.
                - Roses have a long flowering period, ensuring a consistent supply of pollen for crop pollination.
                - They attract a diverse range of pollinators, including butterflies and hoverflies, increasing pollination efficiency.
                - Roses are hardy and adaptable, thriving in various climatic conditions and suitable for different regions.
                - They require low maintenance, being resistant to pests and diseases and tolerating different soil types.
                - Roses add aesthetic value to agricultural landscapes with their vibrant colors and attractive blooms.
                - Roses are a cost-effective option, readily available and easy to propagate, providing an affordable solution to enhance pollination and crop productivity.
                """)
            st.markdown('''
            <style>
            [data-testid="stMarkdownContainer"] ul{
                padding-left:40px;
            }
            </style>
            ''', unsafe_allow_html=True)
        elif prediction == 3 and confidence > 70:
            st.write(f"This is {class_names[3].upper()} and I am  {confidence}% about it")
            st.write("Importance of Sunflowers to Agriculture")
            st.image("SUNFLOWER.jfif")
            st.write("Sunflowers are blablabla")
        elif prediction == 4 and confidence > 70:
            st.write(f"This is {class_names[4].upper()} and I am  {confidence}% about it")
            st.header("Importance of Tulips to Agriculture")
            st.image("TULIP.jfif")
            st.write("Tulips flowers are blablabla")
        else:
            st.image("thinking-think.gif")
            st.write("Sorry I have no idea what type of flower this is. Kindly upload another image of the same flower")

st.header("Developers Note")
st.write("Thank you so much for taking out time to interact with this APP. "
         "\nThis App is intended to solve the little hassle with identifying some specific kind of flowers.")
st.image("1eiE.gif")
st.header('Thank you so very Much!')
