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
    if st.button('Identify this flower'):
        class_names = np.array(['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'])
        predictions = new_model.predict(np.expand_dims(image / 255, 0))
        score = tf.nn.softmax(predictions)
        prediction = np.argmax(score, axis=1)
        confidence = round(np.max(predictions) * 100, 2)
        if prediction == 0 and confidence > 98:
            st.write(f"This is {class_names[0].upper()} and I am  {confidence}% sure about it")
            st.header("Importance of  Daisy Flowers to Agriculture")
            st.image("DAISY.jfif")
            st.markdown(
                """
                These are some of the reasons why you should leave this flower on your field:
                - Abundant blooms: Daisy flowers produce numerous blooms, attracting a large number of pollinators to the farm.
                - Pollen and nectar source: Daisies provide a rich supply of pollen and nectar, serving as an important food source for bees, butterflies, and other pollinators.
                - Generalist attractor: Daisies attract a wide range of pollinators due to their simple, open flower structure and accessible nectar.
                - Long flowering period: Daisies often have an extended blooming season, ensuring a sustained food source for pollinators throughout their active period.
                - Enhancing biodiversity: By attracting diverse pollinators, daisies contribute to increased biodiversity on the farm, supporting the overall ecosystem health.
                - Pollen transfer: Pollinators visiting daisies inadvertently transfer pollen to other agricultural crops, facilitating cross-pollination and promoting seed and fruit set.
                - Improved yield: Effective pollination by daisy-associated pollinators can enhance the yield of neighboring crops by ensuring optimal fertilization and fruit development.
                - Genetic diversity: Cross-pollination with daisies promotes genetic diversity in agricultural crops, which can enhance their resilience to pests, diseases, and environmental stresses.
                - Habitat provision: Daisies serve as habitat and shelter for beneficial insects, providing nesting sites and refuge for pollinators, which can benefit other crops in the vicinity.
                Pest and Diseases
                - Pests: Aphids, Thrips, Spider Mites
                - Diseases: Powdery Mildew, Leaf Spot, Stem Rot
                """
            )
            st.markdown('''
            <style>
            [data-testid="stMarkdownContainer"] ul{
                padding-left:40px;
            }
            </style>
            ''', unsafe_allow_html=True)
        elif prediction == 1 and confidence > 98:
            st.write(f"This is {class_names[1].upper()} and I am  {confidence}% sure about it")
            st.header("Importance of Dandelion Flowers to Agriculture")
            st.image("Dandelions.jfif")
            st.markdown(
                """
                These are some of the reasons why you should leave this flower on your field:
                - Prolific blooming: Dandelions produce numerous flowers, offering abundant opportunities for pollinators to gather nectar and pollen.
                - Early spring resource: Dandelions often bloom early in the spring, providing an important early-season food source for pollinators when other flowers are scarce.
                - Accessibility: Dandelion flowers have an open structure, making it easy for a wide range of pollinators, including bees and butterflies, to access their nectar and pollen.
                - Pollen and nectar production: Dandelions produce ample amounts of pollen and nectar, serving as a valuable food source for pollinators.
                - Long flowering period: Dandelions have a relatively long blooming season, ensuring a sustained supply of nectar and pollen for pollinators throughout their active period.
                - Pollen transfer: Pollinators visiting dandelion flowers inadvertently transfer pollen to other agricultural crops, facilitating cross-pollination and enhancing fruit and seed set.
                - Attracting beneficial insects: Dandelions attract a variety of beneficial insects, such as ladybugs and hoverflies, which can help control pests on the farm.
                - Ecosystem support: Dandelions contribute to the overall biodiversity and ecological balance on the farm by supporting a diverse array of pollinators and other beneficial insects.
                - Resilience and adaptability: Dandelions are hardy plants that can thrive in various environmental conditions, making them reliable sources of pollination services.
                Pest and Diseases
                - Pests: Aphids, Flea Beetles, Leafhoppers
                - Diseases: Rust, Leaf Spot, Dandelion Taraxacum Mosaic Virus
                """
            )
            st.markdown('''
                    <style>
                    [data-testid="stMarkdownContainer"] ul{
                        padding-left:40px;
                    }
                    </style>
                    ''', unsafe_allow_html=True)

        elif prediction == 2 and confidence > 98:
            st.write(f"This is {class_names[2].upper()} and I am  {confidence}% sure about it")
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
                Pests and Diseases
                - Pests: Aphids, Thrips, Japanese Beetles
                - Diseases: Black Spot, Powdery Mildew, Rose Mosaic Virus, Rust
                """)
            st.markdown('''
            <style>
            [data-testid="stMarkdownContainer"] ul{
                padding-left:40px;
            }
            </style>
            ''', unsafe_allow_html=True)
        elif prediction == 3 and confidence > 98:
            st.write(f"This is {class_names[3].upper()} and I am  {confidence}% sure about it")
            st.write("Importance of Sunflowers to Agriculture")
            st.image("SUNFLOWER.jfif")
            st.markdown(
                """
                These are some of the reasons why you should leave this flower on your field:
                - Attractiveness: Sunflowers have bright, showy flowers that attract a wide range of pollinators.
                - Pollen source: The abundant pollen produced by sunflowers serves as a valuable food source for many pollinators.
                - Pollinator diversity: Sunflowers attract diverse pollinators, including bees, butterflies, and even some bird species, which enhances the overall pollination ecosystem on the farm.
                - Extended flowering period: Sunflowers typically have a long blooming season, providing a continuous source of nectar and pollen for pollinators throughout their lifecycle.
                - Pollen transfer: Pollinators visiting sunflowers inadvertently transfer pollen from their flowers to other nearby agricultural crops, promoting cross-pollination.
                - Increased fruit set: Effective pollination by sunflower-attracted pollinators can enhance fruit set and seed production in neighboring crops, leading to higher yields.
                - Genetic diversity: Cross-pollination facilitated by sunflowers contributes to genetic diversity among agricultural crops, which can improve their resilience to pests, diseases, and environmental challenges.
                - Pollination services: Sunflowers act as a hub for pollinators, increasing their presence and activity on the farm, thus benefiting other crops that rely on these pollinators.
                - Enhanced crop quality: Adequate pollination from sunflower-associated pollinators improves the quality attributes of agricultural crops, such as size, shape, color, and taste.
                Pest and Diseases
                - Pests: Aphids, Sunflower Moth, Seed Weevils
                - Diseases: Downy Mildew, Powdery Mildew, Rust
                """
            )
            st.markdown('''
            <style>
            [data-testid="stMarkdownContainer"] ul{
                padding-left:40px;
            }
            </style>
            ''', unsafe_allow_html=True)
        elif prediction == 4 and confidence > 98:
            st.write(f"This is {class_names[4].upper()} and I am  {confidence}% sure about it")
            st.header("Importance of Tulips to Agriculture")
            st.image("TULIP.jfif")
            st.markdown(
                """
                These are some of the reasons why you should leave this flower on your field:
                - Certainly! Here are ten important aspects of tulip flowers in terms of pollination for other agricultural crops:
                - Attraction power: Tulip flowers exhibit vibrant colors and visually appealing patterns that attract a variety of pollinators.
                - Nectar availability: Tulips produce nectar, providing a valuable food source for bees, butterflies, and other pollinating insects.
                - Early blooming: Tulips often bloom early in the growing season, providing an essential nectar and pollen source when other flowers are scarce.
                - Pollinator diversity: Tulips attract a diverse range of pollinators, including bees, butterflies, and hoverflies, enhancing the overall pollination network on the farm.
                - Cross-pollination: Pollinators visiting tulips inadvertently transfer pollen from tulip flowers to other agricultural crops, facilitating cross-pollination and genetic diversity.
                - Extended flowering period: Depending on the variety, tulips can have a relatively long blooming period, ensuring a sustained food source for pollinators throughout their active season.
                - Fruit set improvement: Effective pollination of neighboring crops by tulip-associated pollinators can enhance fruit set and seed production, leading to increased yields.
                - Genetic variability: Cross-pollination with tulips contributes to genetic variability among agricultural crops, which can enhance their adaptability and resilience to changing environmental conditions.
                - Pollination efficiency: Tulip flowers provide easily accessible pollen, which adheres to visiting pollinators, increasing the likelihood of efficient pollen transfer to other crops.
                Pest and Diseases
                - Pests: Aphids, Tulip Bulb Fly, Slugs
                - Diseases: Tulip Fire, Botrytis Blight, Tulip Breaking Virus
                """
            )
            st.markdown('''
            <style>
            [data-testid="stMarkdownContainer"] ul{
                padding-left:40px;
            }
            </style>
            ''', unsafe_allow_html=True)

        else:
            st.image("thinking-think.gif")
            st.write("Sorry I have no idea what type of flower this is. Kindly upload another image of the same flower")
