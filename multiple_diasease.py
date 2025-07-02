import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# ----------------------- Load saved models and scaler -----------------------
diabetes_model = pickle.load(open('diabities_svm.sav', 'rb'))
heart_diseases_model = pickle.load(open('heart_diseases_svm_regreestion.sav', 'rb'))
heart_diseases_scaler = pickle.load(open('heart_diseases_scaler.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinson_svc.sav', 'rb'))

# ----------------------- Prediction Function -----------------------
def predict(model, inputs, scaler=None):
    try:
        inputs = [float(i) for i in inputs]
        inputs = np.array(inputs).reshape(1, -1)
        if scaler:
            inputs = scaler.transform(inputs)
        prediction = model.predict(inputs)
        return prediction[0]
    except ValueError:
        st.error("Please enter valid numeric values in all fields.")
        return None

# ----------------------- Streamlit UI -----------------------
st.set_page_config(page_title="🧠 Multiple Disease Prediction", layout="centered")

# === Sidebar Menu ===
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Home', 'Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['house', 'activity', 'heart', 'person'],
        default_index=0
    )

# ----------------------- Home Page -----------------------
if selected == 'Home':
    st.title("Welcome to the Multiple Disease Prediction System")
    st.write("""
        This application uses machine learning models to predict the presence of:
        - **Diabetes**
        - **Heart Disease**
        - **Parkinson’s Disease**
        
        Please select a test from the sidebar and enter the required medical data.
    """)

# ----------------------- Diabetes Prediction Page -----------------------
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction System')

    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)

        with col1:
            Pregnancies = st.text_input('Number of Pregnancies')
            Glucose = st.text_input('Glucose Level (mg/dL)')
            BloodPressure = st.text_input('Blood Pressure (mm Hg)')
            SkinThickness = st.text_input('Skin Thickness (mm)')

        with col2:
            Insulin = st.text_input('Insulin Level (mu U/ml)')
            BMI = st.text_input('BMI (kg/m²)')
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
            Age = st.text_input('Age (years)')

        submitted = st.form_submit_button('Get Diabetes Result')

        if submitted:
            input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                          Insulin, BMI, DiabetesPedigreeFunction, Age]
            result = predict(diabetes_model, input_data)

            if result == 0:
                st.success('The person is not diabetic.')
            elif result == 1:
                st.error('The person is diabetic.')

# ----------------------- Heart Disease Prediction Page -----------------------
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction System')

    with st.form("heart_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.text_input('Age (years)')
            chol = st.text_input('Cholesterol (mg/dL)')
            exang = st.text_input('Exercise Angina (1 = Yes, 0 = No)')
            thal = st.text_input('Thal (1=Normal, 2=Fixed, 3=Reversible)')

        with col2:
            sex = st.text_input('Gender (1=Male, 0=Female)')
            fbs = st.text_input('Fasting Blood Sugar > 120 mg/dL (1 = Yes, 0 = No)')
            oldpeak = st.text_input('ST Depression (Oldpeak)')
            slope = st.text_input('Slope of ST segment (0–2)')

        with col3:
            cp = st.text_input('Chest Pain Type (0–3)')
            trestbps = st.text_input('Resting Blood Pressure (mm Hg)')
            restecg = st.text_input('Resting ECG (0–2)')
            thalach = st.text_input('Max Heart Rate Achieved')
            ca = st.text_input('Number of Major Vessels (0–3)')

        submitted = st.form_submit_button('Get Heart Disease Result')

        if submitted:
            input_data = [age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]
            result = predict(heart_diseases_model, input_data, scaler=heart_diseases_scaler)

            if result == 0:
                st.success('The person does NOT have a heart disease.')
            elif result == 1:
                st.error('The person HAS a heart disease.')

# ----------------------- Parkinson's Disease Prediction Page -----------------------
if selected == 'Parkinsons Prediction':
    st.title("Parkinson's Disease Prediction System")

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

        submitted = st.form_submit_button('Get Parkinsons Result')

        if submitted:
            input_data = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                          Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR]
            result = predict(parkinsons_model, input_data)

            if result == 0:
                st.success("The person does not have Parkinson’s disease.")
            elif result == 1:
                st.error("The person has Parkinson’s disease.")
