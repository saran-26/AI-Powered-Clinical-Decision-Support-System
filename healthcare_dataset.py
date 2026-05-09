import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000
data = []

for i in range(n):
    # 👤 Patient
    age = np.random.randint(18, 80)
    bmi = np.random.uniform(18, 35)
    bp = np.random.randint(110, 180)
    hr = np.random.randint(60, 120)

    # ⚠️ Condition
    severity = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
    waiting = np.random.randint(5, 200)

    # 👨‍⚕️ Doctor
    doctor_exp = np.random.randint(3, 20)
    success_rate = np.random.randint(70, 95)
    patient_load = np.random.randint(5, 40)

    # 🏥 Hospital
    beds = np.random.randint(20, 100)
    icu = np.random.randint(2, 25)
    equipment = np.random.randint(50, 100)

    # 🎯 STRONG OUTCOME LOGIC (FIXED)
    if severity == 1:
        outcome = "Recovered"

    elif severity == 2:
        if waiting < 60:
            outcome = "Recovered"
        else:
            outcome = "Complication"

    else:  # severity == 3
        if icu < 10:
            outcome = "Deceased"
        else:
            outcome = "Complication"

    # ⏱️ BALANCED TREATMENT TIME (FIXED)
    treatment_time = (
        12
        + (severity * 20)
        + (waiting / 15)
        + (100 - success_rate) / 4
        + (50 - beds) / 8
        + np.random.normal(0, 2)
    )

    data.append([
        i+1, age, round(bmi, 1), bp, hr,
        severity, waiting,
        doctor_exp, success_rate, patient_load,
        beds, icu, equipment,
        outcome, round(treatment_time, 1)
    ])

# 📊 Columns
columns = [
    "patient_id",
    "age",
    "bmi",
    "blood_pressure",
    "heart_rate",
    "severity_level",
    "waiting_time",
    "doctor_experience",
    "doctor_success_rate",
    "doctor_patient_load",
    "bed_availability",
    "icu_capacity",
    "equipment_availability",
    "treatment_outcome",
    "treatment_time"
]

df = pd.DataFrame(data, columns=columns)

# 💾 Save CSV
df.to_csv("healthcare_dataset.csv", index=False)

print("✅ Dataset created successfully!")