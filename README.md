# 🪐 Exoplanet Classifier

**Streamlit app for classifying exoplanet candidates using NASA KOI data and machine learning.**  
Built for the NASA Space Apps Challenge 2025.

## 🚀 Features
- Predict exoplanet likelihood using 3 ML models:
  - Random Forest  
  - Logistic Regression  
  - Gradient Boosting
- Upload single candidates or batch CSVs for analysis
- Visualize light curves interactively with zoom and hover
- Export predictions and track history
- View model confidence, feature importance, and accuracy metrics
- Educational presets (Earth-like, Hot Jupiter) and KOI Score toggle

## 🌐 Live Demo

Try the app here: [nasa-exoplanets.streamlit.app](https://nasa-exoplanets-dj8z8po6dacasd7ygf7qc8.streamlit.app/)

## 🔭 What's Next

- Add support for Kepler and TESS datasets  
- Integrate automatic transit detection  
- Enable user feedback and annotation  
- Expand model suite with deep learning

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/Novachr0no/nasa-exoplanets.git
cd nasa-exoplanets
pip install -r requirements.txt
streamlit run app.py
