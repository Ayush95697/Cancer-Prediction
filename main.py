import streamlit as st
import joblib
import numpy as np

# Load your pre-trained model
model_path = os.path.join("artifacts", "model.joblib")
model = joblib.load(model_path)

st.title("Breast Cancer Diagnosis Prediction")
st.write("Enter the following features to get diagnosis prediction (0 = Benign, 1 = Malignant)")

# Organize inputs into collapsible sections
with st.expander("**Mean Features**", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        radius_mean = st.number_input("Radius Mean", min_value=0.0, value=10.0)
        texture_mean = st.number_input("Texture Mean", min_value=0.0, value=17.0)
        perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, value=100.0)
        area_mean = st.number_input("Area Mean", min_value=0.0, value=700.0)
        smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, value=0.1)
        
    with col2:
        compactness_mean = st.number_input("Compactness Mean", min_value=0.0, value=0.1)
        concavity_mean = st.number_input("Concavity Mean", min_value=0.0, value=0.1)
        concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, value=0.1)
        symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, value=0.1)
        fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, value=0.1)

with st.expander("**Standard Error Features**"):
    col3, col4 = st.columns(2)
    with col3:
        radius_se = st.number_input("Radius SE", min_value=0.0, value=0.5)
        texture_se = st.number_input("Texture SE", min_value=0.0, value=1.0)
        perimeter_se = st.number_input("Perimeter SE", min_value=0.0, value=2.0)
        area_se = st.number_input("Area SE", min_value=0.0, value=50.0)
        smoothness_se = st.number_input("Smoothness SE", min_value=0.0, value=0.01)
        
    with col4:
        compactness_se = st.number_input("Compactness SE", min_value=0.0, value=0.05)
        concavity_se = st.number_input("Concavity SE", min_value=0.0, value=0.05)
        concave_points_se = st.number_input("Concave Points SE", min_value=0.0, value=0.01)
        symmetry_se = st.number_input("Symmetry SE", min_value=0.0, value=0.02)
        fractal_dimension_se = st.number_input("Fractal Dimension SE", min_value=0.0, value=0.01)

with st.expander("**Worst Features**"):
    col5, col6 = st.columns(2)
    with col5:
        radius_worst = st.number_input("Radius Worst", min_value=0.0, value=15.0)
        texture_worst = st.number_input("Texture Worst", min_value=0.0, value=25.0)
        perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0, value=110.0)
        area_worst = st.number_input("Area Worst", min_value=0.0, value=800.0)
        smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0, value=0.15)
        
    with col6:
        compactness_worst = st.number_input("Compactness Worst", min_value=0.0, value=0.3)
        concavity_worst = st.number_input("Concavity Worst", min_value=0.0, value=0.3)
        concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0, value=0.1)
        symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0, value=0.3)
        fractal_dimension_worst = st.number_input("Fractal Dimension Worst", min_value=0.0, value=0.1)

# Create feature array in correct order
features = np.array([[
    # Mean features
    radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
    compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
    fractal_dimension_mean,
    
    # SE features
    radius_se, texture_se, perimeter_se, area_se, smoothness_se,
    compactness_se, concavity_se, concave_points_se, symmetry_se,
    fractal_dimension_se,
    
    # Worst features
    radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
    compactness_worst, concavity_worst, concave_points_worst, symmetry_worst,
    fractal_dimension_worst
]])

# Prediction button
if st.button("Predict Diagnosis"):
    try:
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        
        st.success(f"Prediction: {prediction[0]} (0 = Benign, 1 = Malignant)")
        st.write(f"Confidence: {np.max(probability)*100:.2f}%")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
