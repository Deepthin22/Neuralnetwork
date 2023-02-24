import pandas as pd
from tensorflow import keras
import streamlit as st
from PIL import Image
from keras.models import Sequential 
from keras_preprocessing.sequence import pad_sequences
import pickle

filename = 'heartdisease_model.pkl'
model = pickle.load(open(filename, 'rb'))
##set page configuration
st.set_page_config(page_title='Heart_Disease_classifier',layout='wide')
##add page title and content
st.title('Heart Disease classifier using Artificial neural network')
st.write('please Enter an email to be classified:')

##add image
image=Image.open(r'E:\heart.jpg')
st.image(image,use_column_width=True)
##get user input
age_text=st.number_input('Input age[numeric]:')
anaemia_text=st.number_input('If you are anaemic enter 1 else 0:')
creatinine_text=st.number_input('Input creatinine value[numeric]:')
diabetes_text=st.number_input('If you are diabetic enter 1 else 0:')
ejection_fraction_text=st.number_input('Input ejection_fraction value[numeric]:')
high_blood_pressure_text=st.number_input('If you have high blood pressure enter 1 else 0::')
platelets_text=st.number_input('Input platelets value[numeric]:')
serum_creatinine_text=st.number_input('Input serum_creatinine value:')
serum_sodium_text=st.number_input('Input serum_sodium value [numeric]:')
sex_text=st.number_input('Input your sex 1 if female and 0 if male:')
smoking_text=st.number_input('Input 0 if you are non smoker and 1 if you are smoker:')
time_text=st.number_input('Input time[1 to 300]:')

##convert text to numerical value
##word_index={word:index for index,word in enumerate(df.columns[:-1])}
##numerical_email=[word_index[word] for word in email_text.lower().split() if word in word_index]

##pad the numerical emails so that it can have a unique shape
##padded_email=pad_sequences([numerical_email],maxlen=3000)
##make the prediction
inputv=[[age_text,anaemia_text,creatinine_text,diabetes_text,ejection_fraction_text,high_blood_pressure_text,platelets_text,serum_creatinine_text,serum_sodium_text,sex_text,smoking_text,time_text]]

if st.button('predict'):
    prediction=model.predict(inputv)[0]
    if prediction>0.5:
        st.write('This person has critical heart disease')
    else:
        st.write('This person does not have critical heart disease')