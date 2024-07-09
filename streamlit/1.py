import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

categories = ['Healthy', 'Mild DR', 'Moderate DR', 'Proliferate DR', 'Severe DR']

st.title("Diabetic Retinopathy Detection")

uploaded_file = st.file_uploader("Choose a fundus image...", type=["jpg", "png"])

if uploaded_file is None:
    st.write("Please Upload the fundus image")
else:
    try:
        loaded_model_1 = tf.keras.models.load_model('efb0.h5')
        loaded_model_2 = tf.keras.models.load_model('dn121.h5')

        image_data = uploaded_file.getvalue()

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image_data, caption='Uploaded Image', width=256)

        # Preprocess the image for EfficientNetB0
        img = image.load_img(uploaded_file, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0 

        # Make a prediction using EfficientNetB0
        predictions = loaded_model_1.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = categories[predicted_class_index]

        # Preprocess the image for DenseNet121
        img_2 = image.load_img(uploaded_file, target_size=(224, 224))
        img_array_2 = image.img_to_array(img_2)
        img_array_2 = np.expand_dims(img_array_2, axis=0)
        img_array_2 = img_array_2 / 255.0 

        # Make a prediction using DenseNet121
        predictions_2 = loaded_model_2.predict(img_array_2)
        predicted_class_index_2 = np.argmax(predictions_2)
        predicted_class_2 = categories[predicted_class_index_2]

        background_colors = {
            'Healthy': 'green',
            'Mild DR': 'yellow',
            'Moderate DR': 'orange',
            'Proliferate DR': 'tomato',
            'Severe DR': '#800000'
        }

        # Set the background color based on predicted categories
        bg_color_1 = background_colors.get(predicted_class, 'white')
        bg_color_2 = background_colors.get(predicted_class_2, 'white')

        # Display the predicted category with background color and black text
        st.write(f'<div style="background-color:{bg_color_1}; color:black; padding: 10px; border-radius: 5px;">'
         f'<span style="color:black;">Predicted Category by EfficientNetB0: {predicted_class}</span>'
         '</div>', unsafe_allow_html=True)

        st.write('')  # Add a blank line between the two predictions

        st.write(f'<div style="background-color:{bg_color_2}; color:black; padding: 10px; border-radius: 5px;">'
         f'<span style="color:black;">Predicted Category by DenseNet121: {predicted_class_2}</span>'
         '</div>', unsafe_allow_html=True)


    except FileNotFoundError:
        st.write("Error loading models. Please make sure the model files are present.")
    except Exception as e:
        st.write("An error occurred:", e)
