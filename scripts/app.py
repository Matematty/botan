import os
from PIL import Image
import numpy as np  
from utils import predict_disease,say_disease,resize_and_rescale,speak_text_gtts
import streamlit as st



   

# this is the interface of the app
st.title("Plant Disease Detection System")
st.sidebar.title("About") 
st.write("Upload an image of a plant leaf to detect diseases.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"]) 
process_button = st.button("submit",help="use this button to send ")

if process_button and uploaded_file is not None:
    st.image(uploaded_file,width=100,output_format="auto")
    
    st.spinner("Predicting...")
    prediction = say_disease(predict_disease(uploaded_file)[0],audio_file="audio.mp3")
    st.audio("audio.mp3",format="audio/wav",autoplay=True)

    #st.write(prediction)
    #st.write()
elif process_button and uploaded_file is None:
    st.write("Please upload an image file to proceed.")
  



