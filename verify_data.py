import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Load the created datasets
print("=== LOADING AND VERIFYING DATASETS ===")
patient_history = pd.read_csv('patient_history.csv')
waiting_time = pd.read_csv('waiting_time.csv')
hospital_facilities = pd.read_csv('hospital_facilities.csv')
doctor_details = pd.read_csv('doctor_details.csv')
treatment_outcome = pd.read_csv('treatment_outcome.csv')

# Check basic info
datasets = {
    'Patient History': patient_history,
    'Waiting Time': waiting_time,
    'Hospital Facilities': hospital_facilities,
    'Doctor Details': doctor_details,
    'Treatment Outcome': treatment_outcome
}

for name, df in datasets.items():
    print(f"\n{name}:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    if 'patient_id' in df.columns:
        print(f"Unique patients: {df['patient_id'].nunique()}")

# Check data types and sample
print("\n=== DATA TYPES AND SAMPLE ===")
print(patient_history.info())
print("\nSample data:")
print(patient_history.head(2))