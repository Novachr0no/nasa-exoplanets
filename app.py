import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from sklearn.metrics import classification_report

# === Page config ===
st.set_page_config(page_title="Exoplanet Predictor", page_icon="🚀")

# === Sidebar Control Panel ===
with st.sidebar.expander("📘 Glossary"):
    st.markdown("""
    - **KOI Score**: NASA's confidence in the candidate.
    - **Transit Depth**: How much the star dims during transit.
    - **Planet Radius**: Size compared to Earth.
    - **Orbital Period**: Time it takes to orbit the star.
    - **Transit Duration**: How long the transit lasts.
    - **Exoplanet**: A planet outside our solar system.
    - **Transit Method**: Detecting planets by measuring dips in starlight.
    - **Light Curve**: Graph of brightness vs time, used to spot transits.
    - **False Positive**: A signal that mimics a planet but isn’t one (e.g. binary star).
    - **Machine Learning Model**: Algorithm trained to classify based on patterns in data.
    """)

with st.sidebar.expander("🧠 Model Info"):
    st.markdown(f"""
    **Model Selected**: {st.session_state.get('model_choice', 'N/A')}  
    - Trained on NASA KOI dataset  
    - Uses 5 features: Period, Duration, Depth, Radius, KOI Score  
    - Output: Exoplanet (1) or False Positive (0)  
    """)

with st.sidebar.expander("🌌 Preset Profiles"):
    st.markdown("""
    - **Earth-like**: Small rocky planet, long orbit, shallow transit  
    - **Hot Jupiter**: Large gas giant, short orbit, deep transit  
    """)

with st.sidebar.expander("📂 About This App"):
    st.markdown("""
    This tool uses machine learning to classify exoplanet candidates based on NASA's KOI data.  
    It supports editable inputs, model switching, confidence explanations, and exportable results.
    """)

with st.sidebar.expander("📬 Credits"):
    st.markdown("""
    Built by Amir for NASA Space Apps Challenge 2025  
    GitHub: [github.com/Novachr0no](https://github.com/Novachr0no)  
    """)

with st.sidebar.expander("🌓 Theme Settings"):
    st.markdown("""
    Streamlit supports Light and Dark themes.  
    To switch, click the **Settings gear icon** in the top-right corner and choose your preferred theme.
    """)

# === Session state for history ===
if "history" not in st.session_state:
    st.session_state.history = []

# === Title and instructions ===
st.title("🔭 Exoplanet Classifier")
st.write("Choose a model and enter candidate properties to predict whether it’s an **Exoplanet** or a **False Positive**.")

# === Model selector ===
model_choice = st.selectbox("Choose Model", [
    "Random Forest",
    "Logistic Regression",
    "Gradient Boosting"
])
st.session_state.model_choice = model_choice
model_file = {
    "Random Forest": "random_forest_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "Gradient Boosting": "gradient_boosting_model.pkl"
}[model_choice]
model = joblib.load(model_file)

# === Input Presets ===
preset = st.radio("Choose a preset or enter manually:", ["Manual Entry", "🌍 Earth-like", "☄️ Hot Jupiter"])
koi_unknown = st.checkbox("KOI Score Unknown")

# === Default values based on preset ===
default_values = {
    "Manual Entry": [0.0, 0.0, 0.0, 0.0, 0.5],
    "🌍 Earth-like": [365.25, 10.5, 1500.0, 1.0, 0.9],
    "☄️ Hot Jupiter": [3.5, 2.0, 20000.0, 12.0, 0.95]
}
period_default, duration_default, depth_default, prad_default, score_default = default_values[preset]

# === Editable Inputs ===
period = st.number_input("Orbital Period (days)", value=period_default, step=0.1)
duration = st.number_input("Transit Duration (hours)", value=duration_default, step=0.1)
depth = st.number_input("Transit Depth (ppm)", value=depth_default, step=10.0)
prad = st.number_input("Planet Radius (Earth radii)", value=prad_default, step=0.1)

if koi_unknown:
    score = 0.5
    st.info("KOI Score set to neutral value (0.5) due to unknown status.")
else:
    score = st.slider("KOI Score (0–1)", 0.0, 1.0, score_default, step=0.01)

# === Create dataframe for prediction ===
new_data = pd.DataFrame([{
    'koi_period': period,
    'koi_duration': duration,
    'koi_depth': depth,
    'koi_prad': prad,
    'koi_score': score
}])

# === Prediction ===
if st.button("🔮 Predict"):
    prediction = model.predict(new_data)[0]
    probabilities = model.predict_proba(new_data)[0]
    confidence = max(probabilities)

    if prediction == 1:
        st.success(f"🌍 Likely an Exoplanet! (Confidence: {confidence*100:.2f}%)")
    else:
        st.error(f"❌ Likely a False Positive (Confidence: {confidence*100:.2f}%)")

    st.write("🔎 Prediction Probabilities:")
    st.json({
        "Exoplanet (1)": f"{probabilities[1]*100:.2f}%",
        "False Positive (0)": f"{probabilities[0]*100:.2f}%"
    })

    explanation = ""
    if score > 0.8 and depth > 1000:
        explanation = "Strong signal: high KOI score and deep transit"
    elif score < 0.2 and depth < 100:
        explanation = "Weak signal: low KOI score and shallow transit"
    elif score > 0.5 and prad > 2:
        explanation = "Possible gas giant: moderate score and large radius"
    else:
        explanation = "Mixed signals — further observation needed"

    st.info(f"🧠 Explanation: {explanation}")

    if confidence < 0.6:
        st.warning("⚠️ Prediction confidence is low. Consider further observation or additional data.")

    st.session_state.history.append({
        "Model": model_choice,
        "Orbital Period": period,
        "Transit Duration": duration,
        "Transit Depth": depth,
        "Planet Radius": prad,
        "KOI Score": score if not koi_unknown else "Unknown",
        "Prediction": "Exoplanet" if prediction == 1 else "False Positive",
        "Confidence": f"{confidence*100:.2f}%",
        "Explanation": explanation
    })

# === Light Curve Visualizer ===
st.subheader("🌌 Light Curve Visualizer")
lightcurve_file = st.file_uploader("Upload light curve CSV", type=["csv"], key="lightcurve")

if lightcurve_file is not None:
    lc_df = pd.read_csv(lightcurve_file)
    if "time" in lc_df.columns and "brightness" in lc_df.columns:
        import plotly.express as px
        fig = px.line(lc_df, x="time", y="brightness", title="Light Curve", markers=True)
        fig.update_layout(xaxis_title="Time", yaxis_title="Brightness")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("CSV must contain 'time' and 'brightness' columns.")

# === Batch Prediction ===
st.subheader("📁 Batch Prediction")
batch_file = st.file_uploader("Upload candidate CSV", type=["csv"], key="batch")

if batch_file is not None:
    batch_df = pd.read_csv(batch_file)
    required_cols = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_score']
    if all(col in batch_df.columns for col in required_cols):
        batch_df["Prediction"] = model.predict(batch_df)
        batch_df["Confidence"] = model.predict_proba(batch_df).max(axis=1)

        def explain(row):
            if row["koi_score"] > 0.8 and row["koi_depth"] > 1000:
                return "Strong signal: high KOI score and deep transit"
            elif row["koi_score"] < 0.2 and row["koi_depth"] < 100:
                return "Weak signal: low KOI score and shallow transit"
            elif row["koi_score"] > 0.5 and row["koi_prad"] > 2:
                return "Possible gas giant: moderate score and large radius"
            else:
                return "Mixed signals — further observation needed"

        batch_df["Explanation"] = batch_df.apply(explain, axis=1)
        batch_df["Prediction"] = batch_df["Prediction"].map({1: "Exoplanet", 0: "False Positive"})
        batch_df["Confidence"] = (batch_df["Confidence"] * 100).round(2).astype(str) + "%"

        st.dataframe(batch_df)

        csv_buffer = io.StringIO()
        batch_df.to_csv(csv_buffer, index=False)
        st.download_button("Download Predictions CSV", csv_buffer.getvalue(), "batch_predictions.csv", "text/csv")
    else:
        st.error("CSV must contain columns: koi_period, koi_duration, koi_depth, koi_prad, koi_score")

# === Prediction History ===
if st.session_state.history:
    st.subheader("📜 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)

    st.subheader("📥 Export Prediction History")
    csv_buffer = io.StringIO()
    history_df.to_csv(csv_buffer, index=False)
    st.download_button("Download CSV", csv_buffer.getvalue(), "prediction_history.csv", "text/csv")

    st.subheader("📊 Prediction Distribution")
    st.bar_chart(history_df["Prediction"].value_counts())

# === Feature Importance Viewer ===
st.subheader("📈 Feature Importance")
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    st.bar_chart(pd.Series(importances, index=new_data.columns))
elif hasattr(model, "coef_"):
    coefs = model.coef_[0]
    st.bar_chart(pd.Series(coefs, index=new_data.columns))
else:
    st.info("Feature importance not available for this model.")

# === Model Accuracy Dashboard ===
st.subheader("📊 Model Accuracy Metrics")
try:
    test_data = pd.read_csv("test_data.csv")  # Replace with actual test set if needed
    X_test = test_data.drop(columns=["label"])
    y_test = test_data["label"]
    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    st.json(report)
except Exception as e:
    st.info("Accuracy metrics unavailable. Please provide test data as 'test_data.csv'.")