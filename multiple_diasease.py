import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# load the saved models
diabetes_model = pickle.load(open(r'D:\Internship\SAV_files\diabities_svm.sav','rb'))
heart_diaseases_model = pickle.load(open(r'D:\Internship\SAV_files\heart_diseases_svm_regreestion.sav','rb'))
heart_diaseases_scaler = pickle.load(open(r'D:\Internship\SAV_files\heart_diseases_scaler.sav','rb'))
parkinsons_model = pickle.load(open(r'D:\Internship\SAV_files\parkinson_svc.sav','rb'))

# make a sidebar fro navigation 
with st.sidebar:
       selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )
       
# diabetes prediction page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using SVM')
    
    # input fields for diabetes prediction
    col1, col2 = st.columns(2)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        Glucose = st.text_input('Glucose Level')
        BloodPressure = st.text_input('Blood Pressure Value')
        SkinThickness = st.text_input('Skin Thickness Value')
        
    with col2:
        Insulin = st.text_input('Insulin Level')
        BMI = st.text_input('BMI Value')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
        Age = st.text_input('Age of the Person')

    # prediction button
    if st.button('Diabetes Test Result'):
        input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        input_data_as_float = [float(i) for i in input_data]
        input_data_reshaped = [input_data_as_float]
        
        prediction = diabetes_model.predict(input_data_reshaped)
        
        if prediction[0] == 0:
            st.success('The person is not diabetic.')
        else:
            st.error('The person is diabetic.')

# heart diaseaes
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using SVM')
    
    # input fields for diabetes prediction
    col1, col2,col3 ,col4= st.columns(4)
    
    with col1:
        age = st.text_input('Enter Age')
    with col2:
        sex = st.text_input('Enter Gender (0 = female, 1 = male)')
    with col3:
        cp = st.text_input('Enter Chest Pain Type (cp)')
    with col4:
        trestbps = st.text_input('Enter Resting Blood Pressure (trestbps)')
    with col1:
        chol = st.text_input('Enter Cholesterol Level')
    with col2:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (0 = false; 1 = true)')
    with col3:
        restecg = st.text_input('Resting ECG Results')
    with col4:
        thalach = st.text_input('Maximum Heart Rate Achieved')
    with col1:
        exang = st.text_input('Exercise Induced Angina (0 = no; 1 = yes)')
    with col2:
        oldpeak = st.text_input('ST Depression Induced by Exercise')
    with col3:
        slope = st.text_input('Slope of the Peak Exercise ST Segment')
    with col4:
        ca = st.text_input('Number of Major Vessels Colored by Fluoroscopy (0–3)')
    with col1:
        thal = st.text_input('Thal (1 = normal; 2 = fixed defect; 3 = reversible defect)')

        
    # prediction button
    if st.button('Diabetes Test Result'):
        input_data = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
        input_data_as_float = [float(i) for i in input_data]
        input_data_reshaped = [input_data_as_float]
        input_data_reshaped = heart_diaseases_scaler.transform(input_data_reshaped)
        prediction = heart_diaseases_model.predict(heart_diaseases_scaler.transform(input_data_reshaped))
        if prediction[0] == 0:
            st.success('The person does NOT have a heart disease.')
        else:
            st.error('The person HAS a heart disease.')

# parkinsons prediction page
if selected == 'Parkinsons Prediction':
    st.title('Parkinsons Prediction using SVM')
    
    # input fields for parkinsons prediction
    col1, col2 ,col3 ,col4,col5= st.columns(5)
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        fhi = st.text_input('MDVP:Fhi(Hz)')
        flo = st.text_input('MDVP:Flo(Hz)')
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
    with col2:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        RAP = st.text_input('MDVP:RAP')
        PPQ = st.text_input('MDVP:PPQ')
        DDP = st.text_input('Jitter:DDP')
    with col3:
        Shimmer = st.text_input('MDVP:Shimmer')
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        APQ3 = st.text_input('Shimmer:APQ3')
    with col4:
        APQ5 = st.text_input('Shimmer:APQ5')
        APQ = st.text_input('MDVP:APQ')
        DDA = st.text_input('Shimmer:DDA')
        NHR = st.text_input('NHR')
    with col5:
        HNR = st.text_input('HNR')
    # prediction button
    if st.button('Parkinsons Test Result'):
        input_data = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                      Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR]
        input_data_as_float = [float(i) for i in input_data]
        input_data_reshaped = [input_data_as_float]
        
        prediction = parkinsons_model.predict(input_data_reshaped)
        
        if prediction[0] == 0:
            st.success('The person does not have Parkinsons disease.')
        else:
            st.error('The person has Parkinsons disease.')
