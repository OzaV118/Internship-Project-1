import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Load the saved models and scaler
diabetes_model = pickle.load(open('diabities_svm.sav', 'rb'))
heart_diaseases_model = pickle.load(open('heart_diseases_svm_regreestion.sav', 'rb'))
heart_diaseases_scaler = pickle.load(open('heart_diseases_scaler.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinson_svc.sav', 'rb'))

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction  System')

    with st.form("diabetes_form"):
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

        submitted = st.form_submit_button('Diabetes Test Result')

        if submitted:
            input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                          Insulin, BMI, DiabetesPedigreeFunction, Age]
            input_data_as_float = [float(i) for i in input_data]
            input_data_reshaped = [input_data_as_float]

            prediction = diabetes_model.predict(input_data_reshaped)

            if prediction[0] == 0:
                st.success('The person is not diabetic.')
            else:
                st.error('The person is diabetic.')

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction System')

    with st.form("heart_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.text_input('Age')
            chol = st.text_input('Cholesterol (mg/dL)')
            exang = st.text_input('Exercise Angina (1=Yes, 0=No)')
            thal = st.text_input('Thal (1=Normal, 2=Fixed, 3=Reversible)')

        with col2:
            sex = st.text_input('Gender (0=Female, 1=Male)')
            fbs = st.text_input('Fasting Sugar (1=Yes, 0=No)')
            oldpeak = st.text_input('ST Depression')
            slope = st.text_input('ST Slope (0–2)')

        with col3:
            cp = st.text_input('Chest Pain Type (0–3)')
            trestbps = st.text_input('Resting BP (mm Hg)')
            restecg = st.text_input('Resting ECG (0–2)')
            thalach = st.text_input('Max Heart Rate')
            ca = st.text_input('Major Vessels (0–3)')

        submitted = st.form_submit_button('Heart Disease Test Result')

        if submitted:
            input_data = [age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]
            input_data_as_float = [float(i) for i in input_data]
            input_data_reshaped = [input_data_as_float]
            input_data_scaled = heart_diaseases_scaler.transform(input_data_reshaped)

            prediction = heart_diaseases_model.predict(input_data_scaled)

            if prediction[0] == 0:
                st.success('The person does NOT have a heart disease.')
            else:
                st.error('The person HAS a heart disease.')

# Parkinsons Prediction Page
if selected == 'Parkinsons Prediction':
    st.title('Parkinsons Prediction  System')

    with st.form("parkinsons_form"):
        col1, col2, col3, col4 = st.columns(4)

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
            HNR = st.text_input('HNR')

        with col4:
            APQ5 = st.text_input('Shimmer:APQ5')
            APQ = st.text_input('MDVP:APQ')
            DDA = st.text_input('Shimmer:DDA')
            NHR = st.text_input('NHR')

        
            

        submitted = st.form_submit_button('Parkinsons Test Result')

        if submitted:
            input_data = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                          Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR]
            input_data_as_float = [float(i) for i in input_data]
            input_data_reshaped = [input_data_as_float]

            prediction = parkinsons_model.predict(input_data_reshaped)

            if prediction[0] == 0:
                st.success('The person does not have Parkinson’s disease.')
            else:
                st.error('The person has Parkinson’s disease.')
