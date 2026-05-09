import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

print("🔄 Loading dataset...")

# Load dataset
df = pd.read_csv("healthcare_dataset.csv")

print("✅ Dataset loaded successfully!")
print(df.head())

# ======================
# FEATURES (VERY IMPORTANT)
# ======================
features = [
    'age',
    'bmi',
    'blood_pressure',
    'heart_rate',
    'severity_level',
    'waiting_time',
    'doctor_experience',
    'doctor_success_rate',
    'doctor_patient_load',
    'bed_availability',
    'icu_capacity',
    'equipment_availability'
]

X = df[features]

# ======================
# TARGETS
# ======================
y_class = df['treatment_outcome']
y_reg = df['treatment_time']

# Encode classification target
le = LabelEncoder()
y_class_encoded = le.fit_transform(y_class)

# ======================
# TRAIN-TEST SPLIT
# ======================
X_train, X_test, y_train_c, y_test_c = train_test_split(
    X, y_class_encoded, test_size=0.2, random_state=42
)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# ======================
# TRAIN MODELS
# ======================
print("🤖 Training models...")

# Classification model
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train_c)

# Regression model
reg = RandomForestRegressor(n_estimators=200, random_state=42)
reg.fit(X_train_r, y_train_r)

print("✅ Models trained successfully!")

# ======================
# SAVE MODELS
# ======================
joblib.dump(clf, "best_classifier_enhanced.pkl")
joblib.dump(reg, "xgb_regressor_enhanced.pkl")
joblib.dump(le, "outcome_encoder.pkl")

print("💾 Models saved successfully!")

print("🎯 MODEL READY FOR APP")