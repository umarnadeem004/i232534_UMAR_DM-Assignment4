"""
Part E: Local Front-End Dashboard for Heart Disease Prediction
A Streamlit web app for interactive heart disease risk prediction using XGBoost.

Features:
- 13-input form for patient data
- Real-time prediction with confidence score
- SHAP-based feature importance visualization
- Plain-English risk explanation
- Color-coded risk indicators (green/red)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# Load Models, Scaler, and Metadata
# ============================================================================

@st.cache_resource
def load_model_and_artifacts():
    """Load pre-trained XGBoost model, scaler, and metadata."""
    model_dir = Path('models')
    
    # Load model and scaler
    with open(model_dir / 'best_xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open(model_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load feature names
    with open(model_dir / 'feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    # Load test patient for defaults
    with open(model_dir / 'test_patient.json', 'r') as f:
        test_patient = json.load(f)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    return model, scaler, feature_names, test_patient, explainer


# ============================================================================
# App Configuration
# ============================================================================

st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("❤️ Heart Disease Risk Assessment Dashboard")
st.markdown(
    """
    **Clinical Decision Support System**
    
    This interactive tool uses a machine learning model to assess heart disease risk based on patient vitals and medical history.
    Predictions are based on patterns learned from the UCI Cleveland Heart Disease dataset.
    """
)

# Load artifacts
try:
    model, scaler, feature_names, test_patient, explainer = load_model_and_artifacts()
except FileNotFoundError as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# ============================================================================
# Define Feature Metadata
# ============================================================================

# Feature metadata: input name, display label, min, max, default, description
feature_metadata = {
    'age': {
        'label': 'Age (years)',
        'min': 20.0,
        'max': 80.0,
        'step': 1.0,
        'desc': 'Patient age in years'
    },
    'sex': {
        'label': 'Sex',
        'options': {'Female': 0.0, 'Male': 1.0},
        'desc': 'Biological sex'
    },
    'cp': {
        'label': 'Chest Pain Type',
        'options': {'Typical Angina': 1.0, 'Atypical Angina': 2.0, 'Non-anginal': 3.0, 'Asymptomatic': 4.0},
        'desc': 'Type of chest pain experienced'
    },
    'trestbps': {
        'label': 'Resting Blood Pressure (mmHg)',
        'min': 90.0,
        'max': 200.0,
        'step': 1.0,
        'desc': 'Systolic blood pressure at rest'
    },
    'chol': {
        'label': 'Serum Cholesterol (mg/dL)',
        'min': 100.0,
        'max': 600.0,
        'step': 1.0,
        'desc': 'Total serum cholesterol level'
    },
    'fbs': {
        'label': 'Fasting Blood Sugar > 120 mg/dL?',
        'options': {'No': 0.0, 'Yes': 1.0},
        'desc': 'Whether fasting blood sugar exceeds 120 mg/dL'
    },
    'restecg': {
        'label': 'Resting Electrocardiogram',
        'options': {'Normal': 0.0, 'ST-T Abnormality': 1.0, 'LV Hypertrophy': 2.0},
        'desc': 'Resting ECG results'
    },
    'thalach': {
        'label': 'Maximum Heart Rate Achieved (bpm)',
        'min': 60.0,
        'max': 220.0,
        'step': 1.0,
        'desc': 'Highest heart rate achieved during stress test'
    },
    'exang': {
        'label': 'Exercise Induced Angina?',
        'options': {'No': 0.0, 'Yes': 1.0},
        'desc': 'Whether exercise induces angina'
    },
    'oldpeak': {
        'label': 'ST Depression Induced by Exercise',
        'min': -2.0,
        'max': 10.0,
        'step': 0.1,
        'desc': 'ST depression relative to rest (Exercise induced)'
    },
    'slope': {
        'label': 'Slope of ST Segment',
        'options': {'Upsloping': 1.0, 'Flat': 2.0, 'Downsloping': 3.0},
        'desc': 'Slope of peak exercise ST segment'
    },
    'ca': {
        'label': 'Number of Major Vessels (0-3)',
        'min': 0.0,
        'max': 3.0,
        'step': 1.0,
        'desc': 'Number of major vessels (0-3) colored by fluoroscopy'
    },
    'thal': {
        'label': 'Thalassemia',
        'options': {'Normal': 3.0, 'Fixed Defect': 6.0, 'Reversible Defect': 7.0},
        'desc': 'Thalassemia test result'
    }
}

# ============================================================================
# Sidebar: Input Controls
# ============================================================================

st.sidebar.header("📋 Patient Information")

input_values = {}

# Get defaults from test patient
test_defaults = test_patient['values']
feature_to_idx = {fname: idx for idx, fname in enumerate(feature_names)}

for feature in ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']:
    meta = feature_metadata[feature]
    
    # Get default value from test patient
    default_idx = feature_to_idx.get(feature)
    if default_idx is not None:
        default_value = test_defaults[default_idx]
    else:
        default_value = None
    
    if 'options' in meta:
        # Categorical input
        option_keys = list(meta['options'].keys())
        # Find default option
        default_option_idx = 0
        if default_value is not None:
            for idx, (key, val) in enumerate(meta['options'].items()):
                if val == default_value:
                    default_option_idx = idx
                    break
        
        selected = st.sidebar.selectbox(
            label=meta['label'],
            options=option_keys,
            index=default_option_idx,
            help=meta['desc']
        )
        input_values[feature] = meta['options'][selected]
    else:
        # Continuous input
        if default_value is not None:
            default = float(default_value)
        else:
            default = (meta['min'] + meta['max']) / 2
        
        value = st.sidebar.slider(
            label=meta['label'],
            min_value=meta['min'],
            max_value=meta['max'],
            value=default,
            step=meta['step'],
            help=meta['desc']
        )
        input_values[feature] = value

# Add "Predict" button
st.sidebar.markdown("---")
predict_button = st.sidebar.button("🔍 Predict Risk", use_container_width=True, type="primary")

# ============================================================================
# One-hot encode categorical features
# ============================================================================

def prepare_input_for_model(input_dict, feature_names_list):
    """Convert input dictionary to feature vector matching model's expected format."""
    # Create a row with all features in the correct order
    row_data = []
    
    for fname in feature_names_list:
        if fname in input_dict:
            row_data.append(input_dict[fname])
        elif fname.startswith('cp_') or fname.startswith('restecg_') or fname.startswith('slope_') or fname.startswith('thal_'):
            # One-hot encoded categorical - infer from input values
            cat_prefix = fname.rsplit('_', 1)[0] + '_'
            cat_type = cat_prefix.rstrip('_')
            
            if cat_type == 'cp':
                value = input_dict.get('cp', 1.0)
            elif cat_type == 'restecg':
                value = input_dict.get('restecg', 0.0)
            elif cat_type == 'slope':
                value = input_dict.get('slope', 1.0)
            elif cat_type == 'thal':
                value = input_dict.get('thal', 3.0)
            else:
                value = 0.0
            
            # Check if this category matches the feature
            expected_value = float(fname.split('_')[-1])
            row_data.append(1.0 if value == expected_value else 0.0)
        else:
            row_data.append(0.0)
    
    return np.array([row_data], dtype=np.float32)

# ============================================================================
# Main Prediction Logic
# ============================================================================

if predict_button:
    # Prepare input
    X_input = prepare_input_for_model(input_values, feature_names)
    
    # Make prediction
    prediction_proba = model.predict_proba(X_input)[0]
    prediction_class = model.predict(X_input)[0]
    confidence = max(prediction_proba) * 100
    
    # Compute SHAP values for explanation
    shap_values = explainer.shap_values(X_input)
    
    # ========== Display Results ==========
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Prediction Result")
        
        if prediction_class == 0:
            # No disease
            st.success(f"✅ **LOW RISK** — No Heart Disease Detected")
            risk_color = "green"
            risk_label = "Low Risk"
        else:
            # Disease present
            st.error(f"⚠️ **HIGH RISK** — Heart Disease Likely")
            risk_color = "red"
            risk_label = "High Risk"
        
        # Confidence
        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.1f}%",
            delta=None
        )
        
        # Class probabilities
        prob_no_disease = prediction_proba[0] * 100
        prob_disease = prediction_proba[1] * 100
        
        st.write("**Model Confidence Breakdown:**")
        col_a, col_b = st.columns(2)
        col_a.metric("No Disease", f"{prob_no_disease:.1f}%")
        col_b.metric("Disease Present", f"{prob_disease:.1f}%")
    
    with col2:
        st.subheader("🔍 Key Contributing Factors (Top 3 Features)")
        
        # Extract SHAP values and feature names
        # shap_values is a 2D array (for binary classification, index 1 for positive class)
        if len(shap_values.shape) == 2:
            shap_vals = shap_values[:, 1].flatten()  # Get SHAP values for positive class
        else:
            shap_vals = shap_values.flatten()
        
        # Get absolute SHAP values and sort
        abs_shap = np.abs(shap_vals)
        top_indices = np.argsort(abs_shap)[-3:][::-1]  # Top 3
        
        # Create feature importance dataframe
        top_features_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in top_indices],
            'Importance': [abs_shap[i] for i in top_indices]
        })
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(top_features_df['Feature'], top_features_df['Importance'], color='steelblue')
        ax.set_xlabel('SHAP Importance Score')
        ax.set_title('Top 3 Features Contributing to Prediction')
        ax.invert_yaxis()
        st.pyplot(fig, use_container_width=True)
    
    # ========== Plain-English Explanation ==========
    st.markdown("---")
    st.subheader("📝 Clinical Interpretation")
    
    # Build narrative explanation based on key features and input values
    explanation = generate_explanation(
        prediction_class,
        input_values,
        top_indices,
        feature_names,
        prob_disease
    )
    
    st.markdown(explanation, unsafe_allow_html=True)


# ============================================================================
# Helper: Generate Plain-English Explanation
# ============================================================================

def generate_explanation(prediction_class, inputs, top_feature_indices, feature_names, prob_disease):
    """Generate a plain-English clinical interpretation."""
    
    # Map feature indices to readable names and values
    top_features_info = []
    for idx in top_feature_indices:
        fname = feature_names[idx]
        # Get display name
        if fname.startswith('cp_'):
            display_name = 'Chest Pain Type'
            val = inputs.get('cp', 0)
            cp_types = {1.0: 'Typical Angina', 2.0: 'Atypical Angina', 3.0: 'Non-anginal', 4.0: 'Asymptomatic'}
            val_str = cp_types.get(val, 'Unknown')
        elif fname.startswith('restecg_'):
            display_name = 'Resting ECG'
            val = inputs.get('restecg', 0)
            ecg_types = {0.0: 'Normal', 1.0: 'ST-T Abnormality', 2.0: 'LV Hypertrophy'}
            val_str = ecg_types.get(val, 'Unknown')
        elif fname.startswith('slope_'):
            display_name = 'ST Slope'
            val = inputs.get('slope', 1)
            slope_types = {1.0: 'Upsloping', 2.0: 'Flat', 3.0: 'Downsloping'}
            val_str = slope_types.get(val, 'Unknown')
        elif fname.startswith('thal_'):
            display_name = 'Thalassemia'
            val = inputs.get('thal', 3)
            thal_types = {3.0: 'Normal', 6.0: 'Fixed Defect', 7.0: 'Reversible Defect'}
            val_str = thal_types.get(val, 'Unknown')
        else:
            display_name = fname
            val = inputs.get(fname, 'N/A')
            val_str = f"{val:.1f}" if isinstance(val, (int, float)) else str(val)
        
        top_features_info.append((display_name, val_str))
    
    # Generate narrative
    if prediction_class == 0:
        base = f"This patient presents a **low risk profile** for heart disease (probability: {100-prob_disease:.1f}%). "
    else:
        base = f"This patient presents a **high risk profile** for heart disease (probability: {prob_disease:.1f}%). "
    
    # Add factors
    factors = []
    for i, (fname, val) in enumerate(top_features_info, 1):
        factors.append(f"{i}. **{fname}**: {val}")
    
    factors_text = "\n".join(factors)
    
    if prediction_class == 0:
        conclusion = """
**Recommendation**: Routine cardiovascular monitoring recommended. Continue healthy lifestyle practices 
(regular exercise, balanced diet, stress management). Schedule annual check-ups as per standard clinical guidelines.
        """
    else:
        conclusion = """
**Recommendation**: Further cardiovascular evaluation strongly recommended. Consult with a cardiologist for advanced 
diagnostic testing (e.g., stress test, cardiac catheterization). Consider aggressive management of risk factors and 
medication review.
        """
    
    full_explanation = f"""
{base}

The model identified these as the most influential factors:

{factors_text}

{conclusion}

---
*Note: This tool provides decision support only and should not replace clinical judgment. All predictions should be 
validated by qualified healthcare professionals.*
    """
    
    return full_explanation


# ============================================================================
# Info Section
# ============================================================================

st.markdown("---")
st.markdown("""
### ℹ️ About This Tool

**Model Performance** (Validation Set):
- **ROC-AUC**: 0.941 (Excellent discrimination)
- **Accuracy**: ~83%
- **Sensitivity**: High (detects most disease cases)
- **Specificity**: High (few false alarms)

**Data Source**: UCI Cleveland Heart Disease Database  
**Algorithm**: XGBoost Classifier (Ensemble Learning)  
**Features**: 13 clinical/demographic indicators  
**Sample Size**: 297 patients (297 training, 60 test)  

**Disclaimer**: This tool is for educational and clinical decision support purposes only. 
It should not be used as a substitute for professional medical diagnosis or treatment.
""")
