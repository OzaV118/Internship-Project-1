import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# ------------------- Page Configuration -------------------
st.set_page_config(
    page_title="🧠 Disease Prediction App",
    layout="wide",
    page_icon="🧠"
)

# ------------------- Load Models -------------------
diabetes_model = pickle.load(open('diabities_svm.sav', 'rb'))
heart_diseases_model = pickle.load(open('heart_diseases_svm_regreestion.sav', 'rb'))
heart_diseases_scaler = pickle.load(open('heart_diseases_scaler.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinson_svc.sav', 'rb'))

@st.cache_resource
def load_lung_model():
    try:
        with open("lung_cancer.sav", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Lung cancer model file not found.")
        return None

lung_model = load_lung_model()

# ------------------- Predict Function -------------------
def predict(model, inputs, scaler=None):
    try:
        inputs = np.array([float(i) for i in inputs]).reshape(1, -1)
        if scaler:
            inputs = scaler.transform(inputs)
        return model.predict(inputs)[0]
    except Exception:
        st.error("❌ Please enter valid numeric values in all fields.")
        return None

def predict_lung_risk(inputs):
    data = np.array(inputs).reshape(1, -1)
    result = lung_model.predict(data)[0]
    try:
        confidence = lung_model.predict_proba(data).max()
        return f"{'Low' if result else 'High'} Risk of Lung Cancer (Confidence: {confidence*100:.2f}%)"
    except:
        return f"{'Low' if result else 'High'} Risk of Lung Cancer"

# ------------------- Sidebar Menu -------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3669/3669990.png", width=100)
    selected = option_menu(
        menu_title='Disease Prediction',
        options=['Home', 'Diabetes', 'Heart Disease', "Parkinson's", 'Lung Cancer'],
        icons=['house', 'droplet', 'heart-pulse', 'person-walking', 'lungs'],
        default_index=0
    )

# ------------------- Home -------------------
if selected == 'Home':
    st.title("🧠 Welcome to the Multiple Disease Prediction System")
    st.markdown("""
    ### What this app can do:
    - ✅ Predict **Diabetes**
    - ✅ Predict **Heart Disease**
    - ✅ Predict **Parkinson's Disease**
    - ✅ Predict **Lung Cancer Risk**

    > Built with 💡 Machine Learning & 🖥️ Streamlit
    """)

# ------------------- Diabetes -------------------
elif selected == 'Diabetes':
    st.title('🩸 Diabetes Prediction')
    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)
        with col1:
            Pregnancies = st.text_input('Pregnancies')
            Glucose = st.text_input('Glucose (mg/dL)')
            BloodPressure = st.text_input('Blood Pressure (mm Hg)')
            SkinThickness = st.text_input('Skin Thickness (mm)')
        with col2:
            Insulin = st.text_input('Insulin (mu U/ml)')
            BMI = st.text_input('BMI (kg/m²)')
            DiabetesPedigreeFunction = st.text_input('Pedigree Function')
            Age = st.text_input('Age (years)')
        submit = st.form_submit_button('Predict Diabetes')
        if submit:
            result = predict(diabetes_model, [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
            if result == 1:
                st.error("🛑 Diabetic")
            elif result == 0:
                st.success("✅ Not Diabetic")

# ------------------- Heart Disease -------------------
elif selected == 'Heart Disease':
    st.title('❤️ Heart Disease Prediction')
    with st.form("heart_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.text_input('Age')
            chol = st.text_input('Cholesterol')
            exang = st.text_input('Exercise Induced Angina (1/0)')
            thal = st.text_input('Thal (1-3)')
        with col2:
            sex = st.text_input('Gender (1=Male, 0=Female)')
            fbs = st.text_input('Fasting Blood Sugar > 120 (1/0)')
            oldpeak = st.text_input('ST Depression')
            slope = st.text_input('Slope of ST')
        with col3:
            cp = st.text_input('Chest Pain Type')
            trestbps = st.text_input('Resting BP')
            restecg = st.text_input('ECG (0–2)')
            thalach = st.text_input('Max Heart Rate')
            ca = st.text_input('Major Vessels (0–3)')
        submit = st.form_submit_button('Predict Heart Disease')
        if submit:
            inputs = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            result = predict(heart_diseases_model, inputs, scaler=heart_diseases_scaler)
            if result == 1:
                st.error("🛑 Heart Disease Detected")
            elif result == 0:
                st.success("✅ No Heart Disease")

# ------------------- Parkinson's -------------------
elif selected == "Parkinson's":
    st.title("🧠 Parkinson's Disease Prediction")
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
        submit = st.form_submit_button('Predict Parkinsons')
        if submit:
            inputs = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR]
            result = predict(parkinsons_model, inputs)
            if result == 1:
                st.error("🛑 Parkinson's Disease Detected")
            elif result == 0:
                st.success("✅ No Parkinson's Disease")

# ------------------- Lung Cancer -------------------
elif selected == 'Lung Cancer':
    st.title("🫁 Lung Cancer Risk Prediction")
    gender_map = {'Male': 0, 'Female': 1}
    country_map = {'India': 0, 'USA': 1, 'UK': 2, 'Germany': 3, 'France': 4, 'Italy': 5}
    stage_map = {'Stage I': 0, 'Stage II': 1, 'Stage III': 2, 'Stage IV': 3}
    family_history_map = {'No': 0, 'Yes': 1}
    smoke_map = {'Passive Smoker': 0, 'Never Smoked': 1, 'Former Smoker': 2, 'Current Smoker': 3}
    treatment_map = {'Chemotherapy': 0, 'Surgery': 1, 'Combined': 2, 'Radiation': 3}

    with st.form("lung_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 1, 120, step=1)
            gender = st.selectbox("Gender", list(gender_map.keys()))
            smoker = st.selectbox("Smoking Status", list(smoke_map.keys()))
            fam_history = st.selectbox("Family History", list(family_history_map.keys()))
        with col2:
            country = st.selectbox("Country", list(country_map.keys()))
            stage = st.selectbox("Cancer Stage", list(stage_map.keys()))
            treatment = st.selectbox("Treatment Type", list(treatment_map.keys()))
            bmi = st.number_input("BMI", 10.0, 50.0, step=0.1)
        with col3:
            cholesterol = st.number_input("Cholesterol", 100, 300)
            hypertension = st.radio("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
            asthma = st.radio("Asthma", [0, 1], format_func=lambda x: "Yes" if x else "No")
            cirrhosis = st.radio("Cirrhosis", [0, 1], format_func=lambda x: "Yes" if x else "No")
            other_cancer = st.radio("Other Cancer Diagnosed", [0, 1], format_func=lambda x: "Yes" if x else "No")
        submit = st.form_submit_button("Predict Lung Cancer")

        if submit:
            inputs = [
                age, gender_map[gender], country_map[country], stage_map[stage],
                family_history_map[fam_history], smoke_map[smoker], bmi, cholesterol,
                hypertension, asthma, cirrhosis, other_cancer, treatment_map[treatment]
            ]
            result = predict_lung_risk(inputs)
            st.success(result)
            st.info("This prediction is for informational purposes only.")
