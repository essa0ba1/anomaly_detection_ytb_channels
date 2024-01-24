import joblib 
import pandas as pd 
import numpy as np 
import streamlit as st 


st.set_page_config(page_title=" Aissa Bakhil presents", page_icon="aissa_bakhil2.png", layout="centered")
st.title("Anomaly in Youtube channels") 
st.markdown("""This app based on unsupervised machine learning anomaly detection OneClassSVM 
trained on dataset on 1000 channels ,the anomaly 
detected could be (scam,channel fall , etc or starting channel).""")
data_to_model = []
subs = st.number_input("Number of subscribers ")
views = st.number_input("Total views ")
videos = st.number_input("Number of videos ",)
category = st.selectbox("Channel category ",options=['Music', 'Film & Animation', 'Education', 'Shows', 'Entertainment',
       'Gaming', 'People & Blogs', 'Sports', 'Howto & Style',
       'News & Politics', 'Comedy', 'Trailers', 'Nonprofits & Activism',
       'Science & Technology', 'Movies', 'Pets & Animals',
       'Autos & Vehicles', 'Travel & Events'])
years =[ i for i in range(2005,2025)]
started =  st.selectbox("Channel started ",options=years)

button = st.button("Press to predict")
loaded_label_encoder = joblib.load('models/label_encoder.joblib')
min_max_scaler = joblib.load("models/minmax.joblib")
model = joblib.load("models/oneclass.joblib")
if button:
    category_code = loaded_label_encoder.transform([category])
    data = [subs,views,videos,started,category_code[0]]
    
    scaled = min_max_scaler.transform([data])
    prediction = model.predict(scaled)
    score = model.decision_function(scaled)
    if prediction[0]== -1 : 
        prediction ="Anomaly"
        
    else : 
        prediction= "Normal"
        

    col1, col2= st.columns(2)
    col1.metric("Prediction",str(prediction) )
    col2.metric("Score", str(round(score[0],2)))
   




  
