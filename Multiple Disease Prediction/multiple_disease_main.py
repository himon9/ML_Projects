#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 15:06:05 2023

@author: himon9
"""

import pickle 
import streamlit as st
from streamlit_option_menu import option_menu



# Loading the saved models

# diabetes_model = pickle.load(open("/Users/himon9/Desktop/ML Projects/Multiple Disease Prediction/saved models/diabetes_model.sav",'rb'))

# heart_disease_model = pickle.load(open("/Users/himon9/Desktop/ML Projects/Multiple Disease Prediction/saved models/heart_disease_model.sav",'rb'))

# parkinsons_model = pickle.load(open("/Users/himon9/Desktop/ML Projects/Multiple Disease Prediction/saved models/parkinson_model.sav",'rb'))

# breast_cancer_model = pickle.load(open("/Users/himon9/Desktop/ML Projects/Multiple Disease Prediction/saved models/breast_cancer_model.sav",'rb'))

diabetes_model = pickle.load(open("diabetes_model.sav",'rb'))

heart_disease_model = pickle.load(open("heart_disease_model.sav",'rb'))

parkinsons_model = pickle.load(open("parkinson_model.sav",'rb'))

breast_cancer_model = pickle.load(open("breast_cancer_model.sav",'rb'))



# Sidebar for navigation

with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System ',
                           ['Diabetes Prediction',
                            "Parkinson's Prediction",
                            'Heart Disease Prediction',
                            "Breast Cancer Prediction"],
                           
                           icons=['activity','heart','person'],
                           
                           default_index=0)
    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    # Page title
    st.title('Diabetes Prediction Using ML')
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)
    
    
    
elif (selected == 'Heart Disease Prediction'):
    # Page title
    st.title("Heart Disease Prediction Using ML")

    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
    
    
elif (selected == "Parkinson's Prediction"):
    # Page title
    st.title("Parkinson's Prediction Using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)
elif (selected == "Breast Cancer Prediction"):
    # Page title
    st.title("Breast Cancer Prediction Using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    

    with col1:
        mean_radius = st.text_input('mean_radius')
        
    with col2:
        mean_texture = st.text_input('mean_texture')
        
    with col3:
        mean_perimeter = st.text_input('mean_perimeter')
        
    with col4:
        mean_area = st.text_input('mean_area')
        
    with col5:
        mean_smoothness = st.text_input('mean_smoothness')
        
    with col1:
        mean_compactness = st.text_input('mean_compactness')
        
    with col2:
        mean_concavity = st.text_input('mean_concavity')
        
    with col3:
        mean_concave_points = st.text_input('mean_concave_points')
        
    with col4:
        mean_symmetry = st.text_input('mean_symmetry')
        
    with col5:
        mean_fractal_dimension = st.text_input('mean_fractal_dimension')
        
    with col1:
        radius_error = st.text_input('radius_error')
        
    with col2:
        texture_error = st.text_input('texture_error')
        
    with col3:
        perimeter_error = st.text_input('perimeter_error')
        
    with col4:
        area_error = st.text_input('area_error')
        
    with col5:
        smoothness_error = st.text_input('smoothness_error')
    with col1:
        compactness_error = st.text_input('compactness_error')
        
    with col2:
        concavity_error = st.text_input('concavity_error')
        
    with col3:
        concave_points_error = st.text_input('concave_points_error')
        
    with col4:
        symmetry_error = st.text_input('symmetry_error')
        
    with col5:
        fractal_dimension_error = st.text_input('fractal_dimension_error')
        
    with col1:
        worst_radius = st.text_input('worst_radius')
        
    with col2:
        worst_texture = st.text_input('worst_texture')
        
    with col3:
        worst_perimeter = st.text_input('worst_perimeter')
        
    with col4:
        worst_area = st.text_input('worst_area')
        
    with col5:
        worst_smoothness = st.text_input('worst_smoothness')
        
    with col1:
        worst_compactness = st.text_input('worst_compactness')
        
    with col2:
        worst_concavity = st.text_input('worst_concavity')
        
    with col3:
        worst_concave_points = st.text_input('worst_concave_points')
        
    with col4:
        worst_symmetry = st.text_input('worst_symmetry')
        
    with col5:
        worst_fractal_dimension = st.text_input('worst_fractal_dimension')
        
    

    
    # code for Prediction
    breast_cancer_diagnosis = ''
    # creating a button for Prediction    
    if st.button("Breast Cancer Test Result"):
        breast_cancer_prediction = breast_cancer_model.predict([[mean_radius,
                                                                 mean_texture,
                                                                 mean_perimeter,
                                                                 mean_area,
                                                                 mean_smoothness,
                                                                 mean_compactness,
                                                                 mean_concavity,
                                                                 mean_concave_points,
                                                                 mean_symmetry,
                                                                 mean_fractal_dimension,
                                                                 radius_error,
                                                                 texture_error,
                                                                 perimeter_error,
                                                                 area_error,                 
                                                                 smoothness_error,           
                                                                 compactness_error,          
                                                                 concavity_error,            
                                                                 concave_points_error,      
                                                                 symmetry_error,            
                                                                 fractal_dimension_error,    
                                                                 worst_radius,               
                                                                 worst_texture,              
                                                                 worst_perimeter,            
                                                                 worst_area,                 
                                                                 worst_smoothness,           
                                                                 worst_compactness,         
                                                                 worst_concavity,           
                                                                 worst_concave_points,     
                                                                 worst_symmetry,             
                                                                 worst_fractal_dimension]])                          
        
        if (breast_cancer_prediction[0] == 1):
          breast_cancer_diagnosis = "The person has Breast Cancer"
        else:
          breast_cancer_diagnosis = "The person does not have Breast Cancer"
        
    st.success(breast_cancer_diagnosis)
    
    
    
