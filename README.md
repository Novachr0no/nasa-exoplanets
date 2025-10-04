🪐 Exoplanet Classifier
Streamlit app for classifying exoplanet candidates using NASA KOI data and machine learning.
Built for the NASA Space Apps Challenge 2025.
🚀 Features
- Predict exoplanet likelihood using 7 ML models (RF, LR, GB, SVM, KNN, NB, XGBoost)
- Upload single candidates or batch CSVs for analysis
- Visualize light curves interactively with zoom and hover
- Export predictions and track history
- View model confidence, feature importance, and accuracy metrics
📦 Installation
pip install -r requirements.txt
streamlit run app.py


📁 File Structure
- app.py: Main Streamlit app
- train_model.py: Model training script
- requirements.txt: Dependencies
- koi_data.csv: Training dataset
- test_data.csv: Evaluation set
