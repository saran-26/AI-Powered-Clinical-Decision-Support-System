# =============================================
# STREAMLIT DASHBOARD FOR HEALTHCARE PREDICTION MODEL
# =============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
import base64   # ✅ ADDED

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Clinical Outcome Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ BACKGROUND IMAGE FUNCTION ADDED
def get_base64_of_image(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ✅ LOAD IMAGE
bg_image = get_base64_of_image("bg.jpg")

# Custom CSS (ONLY BACKGROUND ADDED)
st.markdown(f"""
<style>

/* 🔥 BACKGROUND IMAGE */
.stApp {{
    background: linear-gradient(
        rgba(0,0,0,0.75),
        rgba(0,0,0,0.85)
    ),
    url("data:image/jpg;base64,{bg_image}");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* ORIGINAL STYLE */
.main-header {{
    font-size: 4rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}}
.metric-card {{
    background-color: #87CEEB;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
}}
.prediction-card {{
    background-color: #e8f4fd;
    padding: 2rem;
    border-radius: 10px;
    border: 2px solid #1f77b4;
}}

</style>
""", unsafe_allow_html=True)


class ClinicalOutcomePredictionSystem:
    def __init__(self):
        self.load_models()
        
    def load_models(self):
        try:
            self.classifier = joblib.load('best_classifier_enhanced.pkl')
            self.regressor = joblib.load('xgb_regressor_enhanced.pkl')
            self.outcome_encoder = joblib.load('outcome_encoder.pkl')

            self.selected_features = [
                'age','bmi','blood_pressure','heart_rate',
                'severity_level','waiting_time',
                'doctor_experience','doctor_success_rate',
                'doctor_patient_load','bed_availability',
                'icu_capacity','equipment_availability'
            ]

            st.success("✅ Models loaded successfully!")

        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
    
    def create_sidebar_inputs(self):
        st.sidebar.header("🩺 Patient Information")
        
        age = st.sidebar.slider("Age", 1, 100, 45)
        bmi = st.sidebar.slider("BMI", 15.0, 50.0, 25.0)
        blood_pressure = st.sidebar.slider("Blood Pressure", 60, 200, 120)
        heart_rate = st.sidebar.slider("Heart Rate", 40, 150, 75)
        
        severity_level = st.sidebar.selectbox("Severity Level", [1, 2, 3])
        appointment_priority = st.sidebar.selectbox(
            "Appointment Priority", 
            ["Routine", "Follow-up", "Emergency"]
        )
        
        waiting_time = st.sidebar.slider("Waiting Time (minutes)", 0, 480, 60)
        experience_years = st.sidebar.slider("Doctor Experience (years)", 1, 40, 10)
        success_rate = st.sidebar.slider("Doctor Success Rate (%)", 50, 100, 85)
        patient_load = st.sidebar.slider("Doctor Patient Load", 1, 50, 15)
        
        bed_availability = st.sidebar.slider("Bed Availability", 0, 100, 75)
        icu_capacity = st.sidebar.slider("ICU Capacity", 0, 50, 20)
        equipment_availability = st.sidebar.slider("Equipment Availability", 0, 100, 80)
        
        return {
            'age': age,
            'bmi': bmi,
            'blood_pressure': blood_pressure,
            'heart_rate': heart_rate,
            'severity_level': severity_level,
            'appointment_priority': appointment_priority,
            'waiting_time': waiting_time,
            'experience_years': experience_years,
            'success_rate': success_rate,
            'patient_load': patient_load,
            'bed_availability': bed_availability,
            'icu_capacity': icu_capacity,
            'equipment_availability': equipment_availability
        }
    
    def prepare_features(self, input_data):
        data = {
            'age': input_data['age'],
            'bmi': input_data['bmi'],
            'blood_pressure': input_data['blood_pressure'],
            'heart_rate': input_data['heart_rate'],
            'severity_level': input_data['severity_level'],
            'waiting_time': input_data['waiting_time'],
            'doctor_experience': input_data['experience_years'],
            'doctor_success_rate': input_data['success_rate'],
            'doctor_patient_load': input_data['patient_load'],
            'bed_availability': input_data['bed_availability'],
            'icu_capacity': input_data['icu_capacity'],
            'equipment_availability': input_data['equipment_availability']
        }

        df = pd.DataFrame([data])
        df = df[self.selected_features]
        return df
    
    def make_predictions(self, features_df):
        outcome_proba = self.classifier.predict_proba(features_df)[0]
        outcome_pred = self.classifier.predict(features_df)[0]
        outcome_label = self.outcome_encoder.inverse_transform([outcome_pred])[0]
        treatment_time_pred = self.regressor.predict(features_df)[0]
        
        return {
            'outcome': outcome_label,
            'outcome_probabilities': outcome_proba,
            'treatment_time': treatment_time_pred
        }
    
    def display_predictions(self, predictions, input_data):
        st.markdown("## 📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Treatment Outcome")
            st.success(f"{predictions['outcome']}")
        
        with col2:
            st.markdown("### Expected Treatment Time")
            st.info(f"{predictions['treatment_time']:.1f} hours")
        
        with col3:
            st.markdown("### Confidence Level")
            max_prob = max(predictions['outcome_probabilities']) * 100
            st.success(f"{max_prob:.1f}%")
        
        st.markdown("### Outcome Probability Distribution")
        
        fig = px.bar(
            x=self.outcome_encoder.classes_,
            y=predictions['outcome_probabilities'],
            color=predictions['outcome_probabilities'],
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

    def display_analysis_tab(self, input_data):
        st.markdown("## 📊 Patient Risk Analysis")

        col1, col2, col3, col4 = st.columns(4)

        age_risk = "Low" if input_data['age'] < 60 else "High"
        bmi_risk = "Low" if input_data['bmi'] < 25 else "High"
        bp_risk = "Low" if input_data['blood_pressure'] < 130 else "High"
        severity_risk = ["Low","Medium","High"][input_data['severity_level'] - 1]

        col1.metric("Age Risk", age_risk)
        col2.metric("BMI Risk", bmi_risk)
        col3.metric("Blood Pressure Risk", bp_risk)
        col4.metric("Severity Risk", severity_risk)

        st.markdown("### 🔍 Insights")
        st.write(f"""
- Age Risk: {age_risk}  
- BMI Risk: {bmi_risk}  
- Blood Pressure Risk: {bp_risk}  
- Severity Level: {severity_risk}  

👉 Overall risk depends mainly on severity and hospital resources.
        """)

    def display_about_tab(self):
        st.markdown("""
### 🏥 Healthcare Outcome Prediction System

This project presents a **machine learning–based clinical decision support system** designed to assist healthcare professionals in predicting patient treatment outcomes and estimating recovery time.

---

### 🎯 Objectives
- Predict patient outcomes: **Recovered, Complication, Deceased**
- Estimate **treatment duration**
- Provide **risk-based clinical insights**
- Support **data-driven medical decisions**

---

### ⚙️ Methodology
- Data preprocessing and feature engineering  
- Classification using **Supervised Learning models**  
- Regression for treatment time prediction  
- Probability-based prediction for confidence estimation  

---

### 🧠 Technologies Used
- Python  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Pandas, NumPy  

---

### 📊 Key Features
- Real-time prediction system  
- Outcome probability visualization  
- Risk analysis dashboard  
- Interactive UI for patient data input  

---

### 🚀 Application
- Hospital decision support systems  
- Clinical risk assessment  
- Treatment planning optimization  

---

### 📌 Conclusion
This system demonstrates how **machine learning can enhance healthcare efficiency**, reduce uncertainty, and improve patient outcomes through predictive analytics.
        """)

    def run(self):
        st.markdown('<h1 class="main-header">🏥 Clinical Outcome Prediction System</h1>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Analysis", "ℹ️ About"])
        
        input_data = self.create_sidebar_inputs()

        with tab1:
            st.markdown("## Real-time Patient Prediction")
            
            if st.sidebar.button("🚀 Predict", type="primary"):
                features_df = self.prepare_features(input_data)
                predictions = self.make_predictions(features_df)
                self.display_predictions(predictions, input_data)
                self.display_analysis_tab(input_data)
        
        with tab2:
            self.display_analysis_tab(input_data)
        
        with tab3:
            self.display_about_tab()

# Run
if __name__ == "__main__":
    dashboard = ClinicalOutcomePredictionSystem()
    dashboard.run()