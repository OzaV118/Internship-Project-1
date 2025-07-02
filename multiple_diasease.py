import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# ----------------------- Load saved models and scaler -----------------------
diabetes_model = pickle.load(open('diabities_svm.sav', 'rb'))
heart_diseases_model = pickle.load(open('heart_diseases_svm_regreestion.sav', 'rb'))
heart_diseases_scaler = pickle.load(open('heart_diseases_scaler.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinson_svc.sav', 'rb'))

@st.cache_resource
def load_lung_model():
    try:
        with open("lung_cancer.sav", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("The lung cancer model file was not found.")
        return None

lung_model = load_lung_model()

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

def predict_lung_risk(features):
    data = np.array(features).reshape(1, -1)
    result = lung_model.predict(data)[0]

    try:
        confidence = lung_model.predict_proba(data).max()
        confidence_text = f" (Confidence: {confidence*100:.2f}%)"
    except:
        confidence_text = ""

    if result == 0:
        return f"High Risk of Lung Cancer{confidence_text}"
    else:
        return f"Low Risk of Lung Cancer{confidence_text}"

# ----------------------- Streamlit UI -----------------------
st.set_page_config(page_title="🧠 Multiple Disease Prediction", layout="centered")

# === Sidebar Menu ===
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Home', 'Diabetes Prediction', 'Heart Disease Prediction', "Parkinsons Prediction", "Lung Cancer Prediction"],
        icons=['house', 'activity', 'heart', 'person', 'lungs'],
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
        - **Lung Cancer Risk**

        Please select a test from the sidebar and enter the required medical data.
    """)

# ----------------------- Diabetes Prediction Page -----------------------
elif selected == 'Diabetes Prediction':
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
elif selected == 'Heart Disease Prediction':
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

# ----------------------- Parkinson's Prediction Page -----------------------
elif selected == 'Parkinsons Prediction':
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

# ----------------------- Lung Cancer Prediction Page -----------------------
elif selected == 'Lung Cancer Prediction':
    st.title("Lung Cancer Risk Predictor")

    gender_map = {'Male': 0, 'Female': 1}
    country_map = {...}  # same as before
    stage_map = {'Stage III': 0, 'Stage IV': 1, 'Stage I': 2, 'Stage II': 3}
    family_history_map = {'No': 0, 'Yes': 1}
    smoke_map = {'Passive Smoker': 0, 'Never Smoked': 1, 'Former Smoker': 2, 'Current Smoker': 3}
    treatment_map = {'Chemotherapy': 0, 'Surgery': 1, 'Combined': 2, 'Radiation': 3}

    with st.form("lung_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", 1, 120, step=1)
            gender = st.selectbox("Gender", list(gender_map.keys()))
            smoker = st.selectbox("Smoking Status", list(smoke_map.keys()))
            fam_history = st.selectbox("Family History of Cancer?", list(family_history_map.keys()))

        with col2:
            country = st.selectbox("Country", list(country_map.keys()))
            stage = st.selectbox("Cancer Stage", list(stage_map.keys()))
            treatment = st.selectbox("Treatment Type", list(treatment_map.keys()))
            bmi = st.number_input("BMI", 10.0, 50.0, step=0.1)

        with col3:
            cholesterol = st.number_input("Cholesterol Level", 100, 300)
            hypertension = st.radio("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
            asthma = st.radio("Asthma", [0, 1], format_func=lambda x: "Yes" if x else "No")
            cirrhosis = st.radio("Cirrhosis", [0, 1], format_func=lambda x: "Yes" if x else "No")
            other_cancer = st.radio("Other Cancers Diagnosed", [0, 1], format_func=lambda x: "Yes" if x else "No")

        submitted = st.form_submit_button("Get Lung Cancer Risk")

        if submitted:
            if not lung_model:
                st.warning("Lung cancer model not loaded.")
            else:
                inputs = [
                    age, gender_map[gender], country_map[country], stage_map[stage], family_history_map[fam_history],
                    smoke_map[smoker], bmi, cholesterol, hypertension, asthma, cirrhosis, other_cancer,
                    treatment_map[treatment]
                ]
                prediction = predict_lung_risk(inputs)
                st.success(prediction)
                st.info("This is just a prediction. For medical advice, consult a doctor.")
