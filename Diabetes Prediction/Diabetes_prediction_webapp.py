#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 11:02:49 2023

@author: himon9
"""

import numpy as np
import pickle # for loading the saved model
import streamlit as st

# for loading the saved model
loaded_model = pickle.load(open('/Users/himon9/Desktop/ML Projects/Diabetes Prediction/trained_model.sav','rb')) #rb= read binary

# Creating a function for prediction

def diabetes_prediction(input_data):
    #changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0]:
        return "Person is diabetic"
    else:
        return "Person is not diabetic"
    

def main():
    # Giving a title 
    st.title("Diabetes Prediction Web App")
    
    # Getting the input data from the user
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input("Blood Pressure value")
    SkinThickness = st.text_input("Skin Thicknesss value")
    Insulin = st.text_input("Insulin value")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    Age = st.text_input("Age of the Person")
    
    # Code for Prediction
    diagnosis = ''
    
    # Creating a button for prediction
    
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
        
    st.success(diagnosis)
    
    
    
if __name__ == '__main__': # In order to run from Python terminal 
    main()
        
    
    
    

    
    
    
    
    
    
    
    
    