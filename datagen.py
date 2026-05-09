import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate 1000 sample patients
n_patients = 1000

# 🏥 1. PATIENT HISTORY DATASET
patient_history = pd.DataFrame({
    'patient_id': range(1, n_patients + 1),
    'age': np.random.randint(18, 90, n_patients),
    'gender': np.random.choice(['Male', 'Female', 'Other'], n_patients, p=[0.48, 0.50, 0.02]),
    'disease': np.random.choice(['Heart Disease', 'Diabetes', 'Cancer', 'Respiratory', 'Neurological', 
                               'Orthopedic', 'Gastrointestinal', 'Infectious'], n_patients),
    'severity_level': np.random.choice([1, 2, 3], n_patients, p=[0.4, 0.4, 0.2]),
    'previous_conditions': np.random.choice(['0', '1', '2', 'Diabetes, BP', 'Asthma', 'None'], n_patients),
    'previous_treatments': np.random.choice(['0', '1', '2', 'Surgery', 'Medication', 'Therapy'], n_patients),
    'smoking_status': np.random.choice(['Yes', 'No'], n_patients, p=[0.3, 0.7]),
    'bmi': np.round(np.random.uniform(18.5, 35.0, n_patients), 1),
    'blood_pressure': np.random.randint(90, 180, n_patients),
    'heart_rate': np.random.randint(60, 120, n_patients),
    'outcome_history': np.random.choice(['Recovered', 'Re-admitted', 'Complication', 'First Time'], n_patients)
})

# ⏳ 2. WAITING TIME DATASET
waiting_time = pd.DataFrame({
    'patient_id': range(1, n_patients + 1),
    'department': np.random.choice(['Cardiology', 'Neurology', 'Oncology', 'Orthopedics', 
                                  'Emergency', 'General Medicine'], n_patients),
    'waiting_time': np.random.exponential(120, n_patients),  # minutes
    'queue_length': np.random.randint(1, 20, n_patients),
    'appointment_priority': np.random.choice(['Emergency', 'Regular', 'Follow-up'], n_patients, p=[0.2, 0.6, 0.2]),
    'registration_time': [datetime.now() - timedelta(hours=np.random.randint(1, 72)) 
                         for _ in range(n_patients)]
})

# 🏨 3. HOSPITAL FACILITIES DATASET
hospitals = 10
hospital_facilities = pd.DataFrame({
    'hospital_id': range(1, hospitals + 1),
    'hospital_name': [f'Hospital_{i}' for i in range(1, hospitals + 1)],
    'bed_availability': np.random.randint(5, 100, hospitals),
    'icu_capacity': np.random.randint(2, 20, hospitals),
    'equipment_score': np.round(np.random.uniform(3.0, 9.5, hospitals), 1),
    'staff_availability': np.round(np.random.uniform(0.1, 0.8, hospitals), 2),
    'avg_treatment_cost': np.random.randint(5000, 50000, hospitals),
    'specialization': np.random.choice(['Cardiology', 'Neurology', 'Oncology', 'Multi-specialty', 
                                      'Emergency Care'], hospitals)
})

# 👨‍⚕️ 4. DOCTOR DETAILS DATASET
doctors = 50
doctor_details = pd.DataFrame({
    'doctor_id': range(1, doctors + 1),
    'doctor_name': [f'Dr. Doctor_{i}' for i in range(1, doctors + 1)],
    'department': np.random.choice(['Cardiology', 'Neurology', 'Oncology', 'Orthopedics', 
                                  'Emergency', 'General Medicine'], doctors),
    'experience_years': np.random.randint(2, 35, doctors),
    'success_rate': np.round(np.random.uniform(65.0, 95.0, doctors), 1),
    'patient_load': np.random.randint(5, 40, doctors),
    'avg_consult_time': np.random.randint(10, 60, doctors)
})

# 🩺 5. TREATMENT OUTCOME DATASET

treatment_outcome = pd.DataFrame({
    'patient_id': range(1, n_patients + 1),
    'doctor_id': np.random.randint(1, doctors + 1, n_patients),
    'hospital_id': np.random.randint(1, hospitals + 1, n_patients),
})

# 🔗 Merge required data to compute realistic values
temp = treatment_outcome.merge(patient_history, on='patient_id')
temp = temp.merge(waiting_time, on='patient_id')
temp = temp.merge(doctor_details, on='doctor_id')
temp = temp.merge(hospital_facilities, on='hospital_id')

# 🎯 Outcome Logic
outcomes = []
for i in range(len(temp)):
    severity = temp.loc[i, 'severity_level']
    icu = temp.loc[i, 'icu_capacity']
    success = temp.loc[i, 'success_rate']

    if severity == 1 and success > 85:
        outcomes.append('Recovered')
    elif severity == 3 and icu < 8:
        outcomes.append('Deceased')
    else:
        outcomes.append(np.random.choice(['Recovered', 'Complication', 'Re-admitted']))

# ⏱️ Treatment Time Logic (FIXED)
treatment_time = (
    10
    + (temp['severity_level'] * 20)
    + (temp['waiting_time'] / 5)
    + (100 - temp['success_rate']) / 3
    + (50 - temp['bed_availability']) / 5
    + np.random.normal(0, 3, len(temp))
)

# 📦 Add columns back to main dataframe
treatment_outcome['treatment_time'] = treatment_time.round(1)
treatment_outcome['outcome'] = outcomes
treatment_outcome['follow_up_required'] = np.random.choice(['Yes', 'No'], n_patients, p=[0.6, 0.4])

# Save to CSV files
patient_history.to_csv('patient_history.csv', index=False)
waiting_time.to_csv('waiting_time.csv', index=False)
hospital_facilities.to_csv('hospital_facilities.csv', index=False)
doctor_details.to_csv('doctor_details.csv', index=False)
treatment_outcome.to_csv('treatment_outcome.csv', index=False)

print("✅ Sample datasets created successfully!")
print(f"📊 Patient History: {len(patient_history)} records")
print(f"⏳ Waiting Time: {len(waiting_time)} records")
print(f"🏨 Hospital Facilities: {len(hospital_facilities)} records")
print(f"👨‍⚕️ Doctor Details: {len(doctor_details)} records")
print(f"🩺 Treatment Outcome: {len(treatment_outcome)} records")