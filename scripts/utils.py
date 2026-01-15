
from PIL import Image
import streamlit as st
import numpy as np
import tensorflow as tf
import os
import pyttsx3
from gtts import gTTS
from io import BytesIO
from huggingface_hub import hf_hub_download


#base_path = os.path.dirname(__file__)
#model_path  = os.path.join(base_path,"botaniq_model.keras")

REPO_ID = "MateMatty01/BotanIQ_Model"
FILENAME = "botaniq_model.keras"


#from ipynb.fs.full.model import processing_layer

# this the utilities where functions that help the main streamlit interface work

#path = "C:\\Users\\DELL\\Documents\\levels.png"

# this function opens and loads the image from the user devices and passes it to the  resize and rescale function for preprocessing
def load_image(path):

    try:
        img = Image.open(path)
        print("image opened successfully")
        return img
        
    except IOError:
        print("An error occurred while trying to load image")
    except Exception as e:
        print(f"Something unexpected happened while trying to open image {e}")
    
    
    
    

        

# this function resizes and rescales the input image from the user before it is passed to the model
def resize_and_rescale(image_path,target_size=(256,256)):
    # opens the file, checks if there is any error before proceeding
    try:
        
        img = load_image(image_path)

        if img is None:
            return None
        else:
            print("not none")
        

        resized_image = img.resize(target_size)
        #st.write(resized_image.size)
        #print(resized_image.size)
        img_array = np.array(resized_image).astype('float32') / 255.0
        img_array_e = np.expand_dims(img_array,axis=0)
        print("array scaled successfully")

        # returns the resized and rescaled image 
        return img_array_e
        
    except Exception as e:
        print(f"something happened{e}")
        return None
    
    

# this function loads the model from huggingface hub
@st.cache_resource 
def load_model():
    model_path = hf_hub_download(repo_id=REPO_ID,
                                  filename=FILENAME,
                                  token=st.secrets["HF_TOKEN"])
    model = tf.keras.models.load_model(model_path)
    return model 

    
#local_model = tf.keras.models.load_model(model_path) # the local model path
model = load_model() # the huggingface model path

# this function is for predicting the disease 
def predictd_disease(image):
    model = load_model()

   # model = tf.keras.models.load_model(path) # this loads the trained model


    image_to_predict = resize_and_rescale(image) # this preprocesses the image
    #print(image_to_predict)
    # making prediction - checks if the image preprocessing was successful
    if image_to_predict is not None:
        prediction = model.predict(image_to_predict) # this makes the prediction
        print(prediction)

        predicted = np.argmax(prediction,axis=1) # this gets the index of the highest probability
        
        # this returns the prediction made by the model
        return predicted
        
        
    else:
        print("image processing failed")
        return 'prediction not made because image processing failed..'
    


# this function gives our model its voice 
# this uses the pyttsx3 library to convert text to speech
def speak_text(text,file_path):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    #engine.save_to_file(text,filename=file_path)
 

# this function uses the gtts library to convert text to speech
# we would be using the gtts library for its online capabilities bypassing the offline limitations of pyttsx3
def speak_text_gtts(text_to_speak,file_path):
    #text_to_speak = "hello this plant is healthy"
    voice = gTTS(text=text_to_speak, lang='en', slow=False)
    voice_bytes = BytesIO()
    voice.write_to_fp(voice_bytes)

    #os.system("start output.mp3")
    # we need to remove this later but lets test first
    voice.save(file_path)
# this function combines the prediction  and voice feature together
def say_disease(disease_index,audio_file):

    disease_dict = { 
        0: "Tomato Bacterial Spot",
        1: "Tomato Early Blight",
        2: "Tomato Late Blight",
        3: "Tomato Septoria Leaf Spot",
        4: "Healthy Tomato"      
              }
    
    disease_description = {
        0: "Bacterial spot is a common disease in tomatoes caused by the bacterium Xanthomonas campestris pv. vesicatoria. It thrives in warm, wet conditions and spreads through splashing water, contaminated tools, and infected seeds. Symptoms include small, water-soaked spots on leaves that enlarge and turn dark brown or black, often surrounded by a yellow halo. Fruit may also develop raised, scabby lesions. To manage bacterial spot, use disease-free seeds, practice crop rotation, and apply copper-based bactericides as a preventive measure.",
        1: "Early blight is a fungal disease in tomatoes caused by Alternaria solani. It thrives in warm, humid conditions and spreads through infected plant debris, soil, and splashing water. Symptoms include concentric rings on older leaves, leading to yellowing and leaf drop. Fruit may develop dark, sunken lesions. To manage early blight, practice crop rotation, remove infected plant debris, and apply fungicides containing chlorothalonil or copper-based products.",
        2: " Late blight is a devastating disease in tomatoes caused by the oomycete Phytophthora infestans. It thrives in cool, wet conditions and spreads rapidly through airborne spores and water. Symptoms include dark, water-soaked lesions on leaves, stems, and fruit, often with a white fungal growth on the undersides of leaves. Infected fruit may rot completely. To manage late blight, use disease-free seeds, practice crop rotation, and apply fungicides containing chlorothalonil or copper-based products as a preventive measure.",
        3: "Septoria leaf spot is a fungal disease in tomatoes caused by Septoria lycopersici. It thrives in warm, humid conditions and spreads through splashing water and infected plant debris. Symptoms include numerous small, circular spots with dark borders and greyish-white centers, with tiny black specks that may appear inside the spots. To manage septoria leaf spot, remove and destroy all infected leaves immediately and improve airflow by staking the plants. Use fungicides containing Mancozeb or Potassium Bicarbonate as needed.",
        4: "A healthy tomato plant exhibits vibrant green leaves, sturdy stems, and an abundant yield of fruit. The leaves should be free from spots, discoloration, or wilting, indicating the absence of diseases or nutrient deficiencies. The plant should have a strong root system and show consistent growth without any signs of pest infestation. Regular watering, proper fertilization, and adequate sunlight contribute to maintaining the overall health of the tomato plant." }
    
    disease_name = disease_dict.get(disease_index, "Unknown Disease")

    # this is a conditonal to check the model prediction and return the appropriate response
    if disease_dict.get(disease_index) == "Tomato Bacterial Spot":
        text_description = disease_description.get(disease_index)
        text_to_speak = f"The detected disease is {disease_name}, {text_description}"
        st.write(text_to_speak)

        # this calls the speak function 
        speech = speak_text(text_to_speak,file_path=audio_file)
        #speech = speak_text_gtts(text_to_speak,file_path=audio_file)
        #st.write(text_to_speak)
        
        return speech

    elif disease_dict.get(disease_index) == "Tomato Early Blight":
        text_description = disease_description.get(disease_index)
        text_to_speak = f"The detected disease is {disease_name}, {text_description}"  
        st.write(text_to_speak)
        
        # this calls the speak function 
        speech = speak_text(text_to_speak,file_path=audio_file)
        #speech = speak_text_gtts(text_to_speak,file_path=audio_file\)
        #st.write(text_to_speak)
        
        return speech

    elif disease_dict.get(disease_index) == "Tomato Late Blight":

        text_description = disease_description.get(disease_index)
        text_to_speak = f"The detected disease is {disease_name} , {text_description}" 
        st.write(text_to_speak)      
        speech = speak_text(text_to_speak,file_path=audio_file)
        #speech = speak_text_gtts(text_to_speak,file_path=audio_file)
        #st.write(text_to_speak)
        
        return speech
    elif disease_dict.get(disease_index) == "Tomato Septoria Leaf Spot":

        text_description = disease_description.get(disease_index)
        text_to_speak = f"The detected disease is {disease_name} , {text_description}"
        st.write(text_to_speak)
        speech = speak_text(text_to_speak,audio_file)
        #speech = speak_text_gtts(text_to_speak,file_path=audio_file)
        #st.write(text_to_speak)

        return speech
    
    elif disease_dict.get(disease_index) == "Healthy Tomato":
        text_description = disease_description.get(disease_index)
        text_to_speak = f"The plant is healthy with no disease detected ,  {text_description}"
        st.write(text_to_speak)
        speech = speak_text(text_to_speak,file_path=audio_file)
        #speech = speak_text_gtts(text_to_speak,file_path=audio_file)
        return speech
    else:
        text_to_speak = "The detected disease is unknown"
        st.write(text_to_speak)
        speech = speak_text(text_to_speak,file_path=audio_file)
        #speech = speak_text_gtts(text_to_speak,file_path=audio_file)
        return speech
    



    


def main():
    imag = "C:\\Users\\DELL\\Documents\\Plant_Model\\dataset\\train\\Tomato_late_blight\\0a39aa48-3f94-4696-9e43-ff4a93529dc3___RS_Late.B 5103.JPG"
    image_path_ = "C:\\Users\\DELL\\Documents\\Plant_Model\\dataset\\train\\Tomato_late_blight\\0a4b3cde-c83a-4c83-b037-010369738152___RS_Late.B 6985.JPG"
    tomato_spot = "C:\\Users\\DELL\\Documents\\Plant_Model\\dataset\\train\\Tomato_septoria_spot\\0a5edec2-e297-4a25-86fc-78f03772c100___JR_Sept.L.S 8468_180deg.JPG"
    #img = resize_and_rescale(image_path=image_path_,)
    #print(img)
    print(model.summary())
    print(predict_disease(tomato_spot))
    say_disease(predict_disease(tomato_spot)[0],audio_file="audio.mp3")
   # print(load_image(image_path_))
    #print(resize_and_rescale(image_path=image_path_))
    #speak_text("hello this plant is not sick")

"""
# Testing model's actual performance with real image uploads
def predict_tomato_disease(image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    model_v =  tf.keras.models.load_model("botaniq_model.keras")
    predictions = model_v.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    print(f"Predicted Disease: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.2%}")
    print("All probabilities:")

    for i, (name, prob) in enumerate(zip(class_names, predictions[0])):
        print(f"{i+1}. {name}: {prob:.2%}")

    return class_names[predicted_class]


#from google.colab import files
uploaded = files.upload()

for filename in uploaded.keys():
    prediction = predict_tomato_disease(filename)
    print(f"{filename}: {prediction}")
"""
if __name__ == "__main__":
    main()
