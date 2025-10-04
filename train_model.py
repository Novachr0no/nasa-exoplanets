import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# === Load dataset ===
df = pd.read_csv(
    r"C:\Users\Amir\Downloads\cumulative_2025.10.04_02.37.07.csv",
    comment='#'
)

# === Select useful columns ===
needed_cols = ['koi_disposition', 'koi_period', 'koi_duration',
               'koi_depth', 'koi_prad', 'koi_score']
df = df[needed_cols].dropna()

# === Create label column ===
df['label'] = df['koi_disposition'].map({
    'CONFIRMED': 1,
    'FALSE POSITIVE': 0,
    'CANDIDATE': 1
})

# === Features & labels ===
X = df[['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_score']]
y = df['label']

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Train models ===
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
rf_model.fit(X_train, y_train)

log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_model.fit(X_train, y_train)

gb_model = GradientBoostingClassifier(n_estimators=200, random_state=42)
gb_model.fit(X_train, y_train)

# === Save models ===
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(log_model, "logistic_regression_model.pkl")
joblib.dump(gb_model, "gradient_boosting_model.pkl")

print("✅ All models saved successfully.")