import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageOps
from keras.utils.data_utils import get_file


st.title("A Flower Image Recognition App")
st.subheader("A Quick Introduction")
st.image("theo.jfif", width=300)
st.write("Hi, I am Theophilus, a self-taught Data Scientist and a Machine Learning Engineer. "
         "Connect with me via [LinkedIn](https://www.linkedin.com/in/theophilus-chidalu-onyejiaku). "
         "\n You can also Check out my works on [Kaggle](https://www.kaggle.com/theophilusonyejiaku). "
         "Check out [660+ Articles on Python, C++ and R](https://www.educative.io/profile/view/6377211333967872) written by me.")

st.header("Overview")
st.write("This is an  Web App designed to help you identify between 5 different flowers: Tulips, Roses, Sunflower, Daisy and Dandelions")
st.subheader("The Tulips")
st.image("TULIP.jfif", width=300)
st.write("Tulips are a genus of spring-blooming perennial herbaceous bulbiferous geophytes. The flowers are usually large, showy and brightly coloured, generally red, pink, yellow, or white.")

st.subheader("The Sunflower")
st.image("SUNFLOWER.jfif", width=300)
st.write("The common sunflower is a large annual forb of the genus Helianthus grown as a crop for its edible oil and seeds. This sunflower species is also used as wild bird food, as livestock forage, in some industrial applications, and as an ornamental in domestic gardens")


st.subheader("Roses")
st.image('download (2).jfif', width=300)
st.write("A rose is either a woody perennial flowering plant of the genus Rosa, in the family Rosaceae, or the flower it bears. There are over three hundred species and tens of thousands of cultivars")

st.subheader("Daisy")
st.image('DAISY.jfif', width=300)
st.write("The daisy, is a European species of the family Asteraceae, often considered the archetypal species of the name daisy. To distinguish this species from other plants known as daisies,")

st.subheader("Dandelions")
st.image("Dandelions.jfif", width=300)
st.write("Taraxacum is a large genus of flowering plants in the family Asteraceae, which consists of species commonly known as dandelions.")

st.title("Making Predictions")
st.write("""Please upload your choice of flower to predict""")
uploaded_file = st.file_uploader("Choose a jpeg file", type=["jfif", "jpg", "jpeg"])

new_model = load_model("flowerprediction (2).h5")

if uploaded_file is None:
    st.write("Please upload a jpg, jpeg or a jfif image of the flower in the drag and drop box above")
else:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    size = (256, 256)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis, ...]
    st.write("Now click the button below to make prediction")
    if st.button('Make Prediction'):
        class_names = np.array(['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'])
        prediction = new_model.predict(np.expand_dims(image / 255, 0))
        prediction = np.argmax(prediction, axis=1)[:]
        predictions = new_model.predict(np.expand_dims(image / 255, 0))
        score = tf.nn.softmax(predictions[0])
        score = score.numpy()
        score = np.max(score)*100
        if score < 40:
            st.subheader("OOPS!")
            st.image('thinking-think.gif')
            st.subheader("Sorry upload another image of the same flower. My confidence about this prediction is just too low. Better still, upload a better quality jpeg image")
        elif score > 40 and prediction == 0:
            st.subheader(f'This is a {class_names[0]}. I am {round(score)}% confident about this.')
        elif score > 40 and prediction == 1:
            st.subheader(f'This is a {class_names[1]}. I am {round(score)}% confident about this.')
        elif score > 40 and prediction == 2:
            st.subheader(f'This is a {class_names[2]}. I am {round(score)}% confident about this.')
        elif score > 40 and prediction == 3:
            st.subheader(f'This is a {class_names[3]}. I am {round(score)}% confident about this.')
        elif score > 40 and prediction == 4:
            st.subheader(f'This is a {class_names[4]}. I am {round(score)}% confident about this.')


st.header("Developers Note")
st.write("Thank you so much for taking out time to interact with this APP. This App is intended to solve the little hassle with identifying some specific kind of flowers.")
st.image("1eiE.gif")
st.header('Thank you so very Much!')
