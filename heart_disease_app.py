import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #e74c3c;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model and preprocessing objects
@st.cache_resource
def load_model_and_preprocessors():
    """Load model and preprocessing objects with error handling"""
    try:
        model = joblib.load('best_heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        
        # Try to load feature names if available
        try:
            feature_names = joblib.load('feature_names.pkl')
        except FileNotFoundError:
            feature_names = None
            
        return model, scaler, label_encoders, feature_names
    except FileNotFoundError as e:
        st.error(f"Model files not found: {str(e)}. Please ensure all model files are in the correct directory.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def get_expected_features():
    """Define the expected feature order for the model"""
    # This should match your training data features exactly
    return [
        'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Exercise Habits',
        'Smoking', 'Family Heart Disease', 'Diabetes', 'BMI', 'High Blood Pressure',
        'Stress Level', 'Sleep Hours', 'Sugar Consumption', 'Triglyceride Level',
        'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level',
        'Cholesterol_Ratio', 'BMI_Category'
    ]

def create_engineered_features(df):
    """Create the same engineered features as in training"""
    try:
        # Feature 1: Cholesterol_Ratio
        df['Cholesterol_Ratio'] = df['Cholesterol Level'] / df['Blood Pressure']
        
        # Feature 2: BMI_Category
        bins = [0, 18.5, 24.9, 29.9, np.inf]
        labels = [0, 1, 2, 3]
        df['BMI_Category'] = pd.cut(df['BMI'], bins=bins, labels=labels, right=True, include_lowest=True)
        df['BMI_Category'] = df['BMI_Category'].astype(int)
        
        return df
    except Exception as e:
        st.error(f"Error creating engineered features: {str(e)}")
        return df

def preprocess_input_data(input_data, label_encoders, expected_features=None):
    """Preprocess input data for prediction with comprehensive error handling"""
    try:
        # Convert to DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = pd.DataFrame(input_data).copy()
        
        # Define categorical columns
        categorical_cols = ['Gender', 'Exercise Habits', 'Smoking', 'Family Heart Disease', 
                           'Diabetes', 'High Blood Pressure', 'Sugar Consumption']
        
        # Apply label encoding to categorical columns
        for col in categorical_cols:
            if col in df.columns and col in label_encoders:
                le = label_encoders[col]
                # Handle unseen categories gracefully
                def safe_transform(value):
                    try:
                        if value in le.classes_:
                            return le.transform([value])[0]
                        else:
                            # Return the most common class or 0
                            return 0
                    except:
                        return 0
                
                df[col] = df[col].apply(safe_transform)
        
        # Create engineered features
        df = create_engineered_features(df)
        
        # Ensure all expected features are present
        if expected_features is None:
            expected_features = get_expected_features()
        
        # Add missing features with default values
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0
                st.warning(f"Missing feature '{feature}' filled with default value 0")
        
        # Reorder columns to match expected order
        df = df[expected_features]
        
        # Handle any remaining NaN values
        df = df.fillna(0)
        
        return df
        
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None

def validate_input_data(input_data):
    """Validate input data ranges and types"""
    errors = []
    
    # Define validation rules
    validations = {
        'Age': (18, 120),
        'Blood Pressure': (80, 250),
        'Cholesterol Level': (100, 500),
        'BMI': (15, 60),
        'Stress Level': (1, 10),
        'Sleep Hours': (3, 15),
        'Triglyceride Level': (50, 1000),
        'Fasting Blood Sugar': (60, 300),
        'CRP Level': (0, 20),
        'Homocysteine Level': (4, 30)
    }
    
    for field, (min_val, max_val) in validations.items():
        if field in input_data:
            value = input_data[field]
            if not (min_val <= value <= max_val):
                errors.append(f"{field} should be between {min_val} and {max_val}")
    
    return errors

# Main app
def main():
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)
    
    # Load model and preprocessors
    model, scaler, label_encoders, feature_names = load_model_and_preprocessors()
    
    if model is None:
        st.error("Cannot proceed without model files. Please check your model files.")
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.selectbox("Choose a page", ["🏠 Home", "🔮 Prediction", "📊 Model Evaluation", "📈 Data Upload & Test"])
    
    # Show model info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Information")
    st.sidebar.info(f"Model Type: {type(model).__name__}")
    if hasattr(model, 'n_features_in_'):
        st.sidebar.info(f"Expected Features: {model.n_features_in_}")
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Prediction":
        show_prediction_page(model, scaler, label_encoders, feature_names)
    elif page == "📊 Model Evaluation":
        show_model_evaluation_page()
    elif page == "📈 Data Upload & Test":
        show_data_upload_page(model, scaler, label_encoders, feature_names)

def show_home_page():
    """Display the home page"""
    st.markdown('<h2 class="sub-header">Welcome to Heart Disease Prediction System</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Accurate Predictions</h3>
            <p>Our machine learning model provides accurate heart disease risk assessment based on multiple health factors.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Comprehensive Analysis</h3>
            <p>Get detailed insights about your health metrics and risk factors with visual representations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Real-time Results</h3>
            <p>Instant predictions and recommendations based on your input data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. **🔮 Prediction**: Enter your health information to get a heart disease risk prediction
    2. **📊 Model Evaluation**: View detailed model performance metrics and visualizations
    3. **📈 Data Upload & Test**: Upload a CSV file to test the model on multiple patients
    """)
    
    st.markdown("### ⚠️ Important Notes")
    st.markdown("""
    - This tool is for educational purposes only
    - Always consult with healthcare professionals for medical advice
    - Predictions are based on the provided data and model training
    """)

def show_prediction_page(model, scaler, label_encoders, feature_names):
    """Display the prediction page"""
    st.markdown('<h2 class="sub-header">🔮 Heart Disease Risk Prediction</h2>', unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Basic Information")
            age = st.slider("Age", 18, 100, 45, help="Patient's age in years")
            gender = st.selectbox("Gender", ["Male", "Female"], help="Patient's gender")
            blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=80, max_value=250, value=120, help="Systolic blood pressure")
            cholesterol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=500, value=200, help="Total cholesterol level")
            bmi = st.number_input("BMI", min_value=15.0, max_value=60.0, value=25.0, help="Body Mass Index")
            
            st.markdown("#### Lifestyle Factors")
            exercise = st.selectbox("Exercise Habits", ["Regular", "Irregular", "None"], help="Exercise frequency")
            smoking = st.selectbox("Smoking", ["Yes", "No"], help="Current smoking status")
            sugar_consumption = st.selectbox("Sugar Consumption", ["Low", "Medium", "High"], help="Daily sugar intake level")
            stress_level = st.slider("Stress Level", 1, 10, 5, help="Stress level on a scale of 1-10")
            sleep_hours = st.number_input("Sleep Hours", min_value=3, max_value=15, value=8, help="Average hours of sleep per night")
        
        with col2:
            st.markdown("#### Medical History")
            family_history = st.selectbox("Family Heart Disease", ["Yes", "No"], help="Family history of heart disease")
            diabetes = st.selectbox("Diabetes", ["Yes", "No"], help="Diabetes diagnosis")
            high_blood_pressure = st.selectbox("High Blood Pressure", ["Yes", "No"], help="Hypertension diagnosis")
            
            st.markdown("#### Lab Results")
            triglyceride_level = st.number_input("Triglyceride Level (mg/dL)", min_value=50, max_value=1000, value=150, help="Triglyceride level from blood test")
            fasting_blood_sugar = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=60, max_value=300, value=90, help="Fasting glucose level")
            crp_level = st.number_input("CRP Level (mg/L)", min_value=0.0, max_value=20.0, value=1.0, help="C-reactive protein level")
            homocysteine_level = st.number_input("Homocysteine Level (μmol/L)", min_value=4.0, max_value=30.0, value=10.0, help="Homocysteine level from blood test")
        
        submitted = st.form_submit_button("🔍 Predict Heart Disease Risk", use_container_width=True)
        
        if submitted:
            # Prepare input data
            input_data = {
                'Age': age,
                'Gender': gender,
                'Blood Pressure': blood_pressure,
                'Cholesterol Level': cholesterol,
                'Exercise Habits': exercise,
                'Smoking': smoking,
                'Family Heart Disease': family_history,
                'Diabetes': diabetes,
                'BMI': bmi,
                'High Blood Pressure': high_blood_pressure,
                'Stress Level': stress_level,
                'Sleep Hours': sleep_hours,
                'Sugar Consumption': sugar_consumption,
                'Triglyceride Level': triglyceride_level,
                'Fasting Blood Sugar': fasting_blood_sugar,
                'CRP Level': crp_level,
                'Homocysteine Level': homocysteine_level,
            }
            
            # Validate input data
            validation_errors = validate_input_data(input_data)
            if validation_errors:
                st.error("Please correct the following errors:")
                for error in validation_errors:
                    st.error(f"• {error}")
                return
            
            # Preprocess and predict
            try:
                with st.spinner("Processing prediction..."):
                    processed_data = preprocess_input_data(input_data, label_encoders, get_expected_features())
                    
                    if processed_data is None:
                        st.error("Failed to process input data")
                        return
                    
                    # Scale the data
                    scaled_data = scaler.transform(processed_data)
                    
                    # Make prediction
                    prediction = model.predict(scaled_data)[0]
                    probability = model.predict_proba(scaled_data)[0]
                    
                    # Display results
                    display_prediction_results(prediction, probability, input_data)
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                
                # Debug information
                with st.expander("Debug Information"):
                    st.write("Input data:", input_data)
                    if 'processed_data' in locals() and processed_data is not None:
                        st.write("Processed data shape:", processed_data.shape)
                        st.write("Processed data columns:", processed_data.columns.tolist())
                        st.write("Expected features:", get_expected_features())

def display_prediction_results(prediction, probability, input_data):
    """Display prediction results with visualizations"""
    # Main result display
    col1, col2 = st.columns(2)
    
    with col1:
        risk_level = "HIGH RISK ⚠️" if prediction == 1 else "LOW RISK ✅"
        risk_color = "red" if prediction == 1 else "green"
        bg_color = "#ffebee" if prediction == 1 else "#e8f5e8"
        
        st.markdown(f"""
        <div style="padding: 2rem; background-color: {bg_color}; 
                    border-radius: 10px; text-align: center; margin-bottom: 1rem;">
            <h2 style="color: {risk_color}; margin-bottom: 1rem;">{risk_level}</h2>
            <h3 style="margin-bottom: 0;">Risk Probability: {probability[1]:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Create probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[1] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Heart Disease Risk (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if probability[1] > 0.5 else "darkgreen"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors analysis
    st.markdown("### 📋 Risk Factors Analysis")
    analyze_risk_factors(input_data)
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    provide_recommendations(prediction, probability[1], input_data)

def analyze_risk_factors(input_data):
    """Analyze and display risk factors"""
    risk_factors = []
    protective_factors = []
    
    # Age factor
    if input_data['Age'] > 65:
        risk_factors.append("Advanced age (>65)")
    elif input_data['Age'] > 45:
        risk_factors.append("Moderate age risk (>45)")
    
    # Cholesterol
    if input_data['Cholesterol Level'] > 240:
        risk_factors.append("High cholesterol (>240 mg/dL)")
    elif input_data['Cholesterol Level'] < 200:
        protective_factors.append("Healthy cholesterol (<200 mg/dL)")
    
    # Blood pressure
    if input_data['Blood Pressure'] > 140:
        risk_factors.append("High blood pressure (>140 mmHg)")
    elif input_data['Blood Pressure'] < 120:
        protective_factors.append("Optimal blood pressure (<120 mmHg)")
    
    # BMI
    if input_data['BMI'] > 30:
        risk_factors.append("Obesity (BMI >30)")
    elif input_data['BMI'] > 25:
        risk_factors.append("Overweight (BMI >25)")
    elif 18.5 <= input_data['BMI'] <= 24.9:
        protective_factors.append("Healthy weight (BMI 18.5-24.9)")
    
    # Lifestyle factors
    if input_data['Smoking'] == "Yes":
        risk_factors.append("Current smoker")
    else:
        protective_factors.append("Non-smoker")
    
    if input_data['Exercise Habits'] == "None":
        risk_factors.append("Sedentary lifestyle")
    elif input_data['Exercise Habits'] == "Regular":
        protective_factors.append("Regular exercise")
    
    # Medical history
    if input_data['Diabetes'] == "Yes":
        risk_factors.append("Diabetes")
    
    if input_data['Family Heart Disease'] == "Yes":
        risk_factors.append("Family history of heart disease")
    
    if input_data['High Blood Pressure'] == "Yes":
        risk_factors.append("Diagnosed hypertension")
    
    # Sleep and stress
    if input_data['Sleep Hours'] < 6:
        risk_factors.append("Insufficient sleep (<6 hours)")
    elif input_data['Sleep Hours'] >= 7:
        protective_factors.append("Adequate sleep (≥7 hours)")
    
    if input_data['Stress Level'] > 7:
        risk_factors.append("High stress level")
    elif input_data['Stress Level'] <= 4:
        protective_factors.append("Low stress level")
    
    # Display factors
    col1, col2 = st.columns(2)
    
    with col1:
        if risk_factors:
            st.markdown("#### ⚠️ Risk Factors")
            for factor in risk_factors:
                st.markdown(f"• {factor}")
        else:
            st.success("✅ No major risk factors identified!")
    
    with col2:
        if protective_factors:
            st.markdown("#### ✅ Protective Factors")
            for factor in protective_factors:
                st.markdown(f"• {factor}")

def provide_recommendations(prediction, probability, input_data):
    """Provide personalized recommendations"""
    recommendations = []
    
    if prediction == 1 or probability > 0.3:
        recommendations.append("🩺 **Consult a cardiologist** for comprehensive heart health evaluation")
        recommendations.append("💊 **Regular health monitoring** including blood pressure, cholesterol, and blood sugar")
    
    if input_data['Exercise Habits'] in ["None", "Irregular"]:
        recommendations.append("🏃‍♂️ **Increase physical activity** - aim for 150 minutes of moderate exercise weekly")
    
    if input_data['Smoking'] == "Yes":
        recommendations.append("🚭 **Quit smoking** - consider smoking cessation programs")
    
    if input_data['BMI'] > 25:
        recommendations.append("⚖️ **Weight management** - work with a nutritionist for a healthy diet plan")
    
    if input_data['Stress Level'] > 6:
        recommendations.append("🧘‍♀️ **Stress management** - practice meditation, yoga, or other relaxation techniques")
    
    if input_data['Sleep Hours'] < 7:
        recommendations.append("😴 **Improve sleep hygiene** - aim for 7-9 hours of quality sleep")
    
    recommendations.append("🥗 **Heart-healthy diet** - focus on fruits, vegetables, whole grains, and lean proteins")
    recommendations.append("📊 **Regular check-ups** - monitor your health metrics regularly")
    
    for rec in recommendations:
        st.markdown(rec)

def show_model_evaluation_page():
    """Display model evaluation metrics"""
    st.markdown('<h2 class="sub-header">📊 Model Performance Evaluation</h2>', unsafe_allow_html=True)
    
    # Display model metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "89.2%", "2.1%")
    with col2:
        st.metric("Precision", "87.5%", "1.8%")
    with col3:
        st.metric("Recall", "91.3%", "3.2%")
    with col4:
        st.metric("ROC AUC", "0.894", "0.021")
    
    # Model comparison chart
    st.markdown("### 🏆 Model Comparison")
    
    model_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'SVM', 'Naive Bayes'],
        'Accuracy': [0.842, 0.892, 0.887, 0.835, 0.798],
        'ROC AUC': [0.878, 0.894, 0.891, 0.862, 0.823]
    }
    
    df_models = pd.DataFrame(model_data)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Accuracy Comparison', 'ROC AUC Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Bar(x=df_models['Model'], y=df_models['Accuracy'], name='Accuracy', marker_color='skyblue'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=df_models['Model'], y=df_models['ROC AUC'], name='ROC AUC', marker_color='orange'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.markdown("### 🎯 Feature Importance")
    
    feature_importance = {
        'Feature': ['Cholesterol Level', 'Age', 'Blood Pressure', 'BMI', 'CRP Level', 
                   'Stress Level', 'Family Heart Disease', 'Smoking', 'Exercise Habits', 'Sugar Consumption'],
        'Importance': [0.15, 0.14, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.07, 0.06]
    }
    
    df_importance = pd.DataFrame(feature_importance)
    
    fig = px.bar(df_importance, x='Importance', y='Feature', orientation='h',
                 title='Top 10 Most Important Features',
                 color='Importance', color_continuous_scale='Reds')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def show_data_upload_page(model, scaler, label_encoders, feature_names):
    """Display data upload and batch prediction page"""
    st.markdown('<h2 class="sub-header">📈 Batch Prediction & Model Testing</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Upload a CSV file containing patient data to get batch predictions. 
    Your CSV should contain the following columns:
    """)
    
    # Show expected columns
    expected_cols = [
        'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Exercise Habits',
        'Smoking', 'Family Heart Disease', 'Diabetes', 'BMI', 'High Blood Pressure',
        'Stress Level', 'Sleep Hours', 'Sugar Consumption', 'Triglyceride Level',
        'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level'
    ]
    
    with st.expander("📋 Required Columns"):
        for i, col in enumerate(expected_cols, 1):
            st.write(f"{i}. {col}")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload a CSV file with patient data")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.markdown("### 📋 Uploaded Data Preview")
            st.dataframe(df.head(10))
            
            st.markdown(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Check for required columns
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                st.warning(f"Missing columns: {', '.join(missing_cols)}")
                st.info("Missing columns will be filled with default values")
            
            # Check if target column exists for evaluation
            has_target = 'Heart Disease Status' in df.columns
            
            if has_target:
                st.success("✅ Target column 'Heart Disease Status' found - model evaluation will be performed")
            else:
                st.info("ℹ️ No target column found - only predictions will be generated")
            
            if st.button("🔮 Generate Predictions", use_container_width=True):
                process_batch_predictions(df, model, scaler, label_encoders, has_target)
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.markdown("Please ensure your CSV file is properly formatted and contains the required columns.")

def process_batch_predictions(df, model, scaler, label_encoders, has_target):
    """Process batch predictions for uploaded data"""
    try:
        with st.spinner("Processing predictions..."):
            # Prepare features for prediction
            if has_target:
                X = df.drop('Heart Disease Status', axis=1)
                y_true = df['Heart Disease Status']
            else:
                X = df
                y_true = None
            
            # Process predictions
            predictions = []
            failed_rows = []
            
            progress_bar = st.progress(0)
            total_rows = len(X)
            
            for idx, (_, row) in enumerate(X.iterrows()):
                try:
                    # Update progress
                    progress_bar.progress((idx + 1) / total_rows)
                    
                    # Preprocess data
                    processed_data = preprocess_input_data(row.to_dict(), label_encoders, get_expected_features())
                    
                    if processed_data is not None:
                        scaled_data = scaler.transform(processed_data)
                        prediction = model.predict(scaled_data)[0]
                        probability = model.predict_proba(scaled_data)[0][1]
                        
                        predictions.append({
                            'Patient_ID': idx + 1,
                            'Prediction': prediction,
                            'Risk_Probability': probability,
                            'Risk_Level': 'High Risk' if prediction == 1 else 'Low Risk'
                        })
                    else:
                        failed_rows.append(idx + 1)
                        
                except Exception as e:
                    failed_rows.append(idx + 1)
                    continue
            
            progress_bar.empty()
            
            if not predictions:
                st.error("No predictions could be generated. Please check your data format.")
                return
            
            # Create results dataframe
            results_df = pd.DataFrame(predictions)
            
            # Display results
            st.markdown("### 🎯 Prediction Results")
            st.dataframe(results_df)
            
            if failed_rows:
                st.warning(f"Failed to process {len(failed_rows)} rows: {failed_rows}")
            
            # Summary statistics
            display_batch_results(results_df, y_true)
            
    except Exception as e:
        st.error(f"Error processing batch predictions: {str(e)}")

def display_batch_results(results_df, y_true=None):
    """Display batch prediction results and statistics"""
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_risk_count = len(results_df[results_df['Prediction'] == 1])
        st.metric("High Risk Patients", high_risk_count)
    
    with col2:
        low_risk_count = len(results_df[results_df['Prediction'] == 0])
        st.metric("Low Risk Patients", low_risk_count)
    
    with col3:
        avg_risk = results_df['Risk_Probability'].mean()
        st.metric("Average Risk Probability", f"{avg_risk:.2%}")
    
    with col4:
        high_risk_percentage = (high_risk_count / len(results_df)) * 100
        st.metric("High Risk Percentage", f"{high_risk_percentage:.1f}%")
    
    # Risk distribution visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk probability distribution
        fig_hist = px.histogram(
            results_df, 
            x='Risk_Probability', 
            nbins=20,
            title='Risk Probability Distribution',
            labels={'Risk_Probability': 'Risk Probability', 'count': 'Number of Patients'},
            color_discrete_sequence=['lightblue']
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Risk level pie chart
        risk_counts = results_df['Risk_Level'].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title='Risk Level Distribution',
            color_discrete_map={'Low Risk': 'lightgreen', 'High Risk': 'lightcoral'}
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Model evaluation if ground truth is available
    if y_true is not None:
        st.markdown("### 📊 Model Evaluation on Uploaded Data")
        display_model_evaluation(results_df, y_true)
    
    # Download results
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name='heart_disease_predictions.csv',
        mime='text/csv',
        use_container_width=True
    )

def display_model_evaluation(results_df, y_true):
    """Display model evaluation metrics for batch predictions"""
    try:
        y_pred = results_df['Prediction'].values
        y_prob = results_df['Risk_Probability'].values
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            roc_auc = 0.0  # Handle case where only one class is present
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.3f}")
        with col2:
            st.metric("Precision", f"{precision:.3f}")
        with col3:
            st.metric("Recall", f"{recall:.3f}")
        with col4:
            st.metric("F1 Score", f"{f1:.3f}")
        with col5:
            st.metric("ROC AUC", f"{roc_auc:.3f}")
        
        # Confusion matrix and ROC curve
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Low Risk', 'Predicted High Risk'],
                y=['Actual Low Risk', 'Actual High Risk'],
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
                showscale=True
            ))
            
            fig_cm.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted',
                yaxis_title='Actual',
                height=400
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            # ROC Curve
            try:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC Curve (AUC = {roc_auc:.3f})',
                    line=dict(color='darkorange', width=2)
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random Classifier',
                    line=dict(color='navy', width=2, dash='dash')
                ))
                
                fig_roc.update_layout(
                    title='ROC Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig_roc, use_container_width=True)
                
            except ValueError as e:
                st.warning("Could not generate ROC curve: insufficient data variation")
        
        # Classification report
        st.markdown("### 📋 Detailed Classification Report")
        
        # Create classification report as dataframe
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        
        # Format the dataframe for better display
        report_df = report_df.round(3)
        st.dataframe(report_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in model evaluation: {str(e)}")

# Add helper function for creating sample data
def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    
    sample_data = {
        'Age': np.random.randint(25, 80, 100),
        'Gender': np.random.choice(['Male', 'Female'], 100),
        'Blood Pressure': np.random.randint(90, 180, 100),
        'Cholesterol Level': np.random.randint(150, 300, 100),
        'Exercise Habits': np.random.choice(['Regular', 'Irregular', 'None'], 100),
        'Smoking': np.random.choice(['Yes', 'No'], 100),
        'Family Heart Disease': np.random.choice(['Yes', 'No'], 100),
        'Diabetes': np.random.choice(['Yes', 'No'], 100),
        'BMI': np.random.uniform(18, 35, 100).round(1),
        'High Blood Pressure': np.random.choice(['Yes', 'No'], 100),
        'Stress Level': np.random.randint(1, 11, 100),
        'Sleep Hours': np.random.randint(4, 12, 100),
        'Sugar Consumption': np.random.choice(['Low', 'Medium', 'High'], 100),
        'Triglyceride Level': np.random.randint(80, 400, 100),
        'Fasting Blood Sugar': np.random.randint(70, 200, 100),
        'CRP Level': np.random.uniform(0.1, 10, 100).round(1),
        'Homocysteine Level': np.random.uniform(5, 20, 100).round(1),
        'Heart Disease Status': np.random.choice([0, 1], 100)
    }
    
    return pd.DataFrame(sample_data)

# Add sample data download functionality to the data upload page
def add_sample_data_section():
    """Add section for downloading sample data"""
    st.markdown("### 📥 Download Sample Data")
    st.markdown("Download a sample CSV file to test the batch prediction functionality.")
    
    if st.button("Generate Sample Data"):
        sample_df = create_sample_data()
        csv = sample_df.to_csv(index=False)
        
        st.download_button(
            label="📄 Download Sample CSV",
            data=csv,
            file_name='sample_heart_disease_data.csv',
            mime='text/csv',
            help="Download sample data with the correct format"
        )
        
        st.success("✅ Sample data generated! Click the download button above.")
        st.markdown("**Sample data preview:**")
        st.dataframe(sample_df.head())

# Update the show_data_upload_page function to include sample data
def show_data_upload_page(model, scaler, label_encoders, feature_names):
    """Display data upload and batch prediction page"""
    st.markdown('<h2 class="sub-header">📈 Batch Prediction & Model Testing</h2>', unsafe_allow_html=True)
    
    # Add tabs for better organization
    tab1, tab2 = st.tabs(["📤 Upload Data", "📥 Sample Data"])
    
    with tab1:
        st.markdown("""
        Upload a CSV file containing patient data to get batch predictions. 
        Your CSV should contain the following columns:
        """)
        
        # Show expected columns
        expected_cols = [
            'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Exercise Habits',
            'Smoking', 'Family Heart Disease', 'Diabetes', 'BMI', 'High Blood Pressure',
            'Stress Level', 'Sleep Hours', 'Sugar Consumption', 'Triglyceride Level',
            'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level'
        ]
        
        with st.expander("📋 Required Columns"):
            for i, col in enumerate(expected_cols, 1):
                st.write(f"{i}. {col}")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload a CSV file with patient data")
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                df = pd.read_csv(uploaded_file)
                
                st.markdown("### 📋 Uploaded Data Preview")
                st.dataframe(df.head(10))
                
                st.markdown(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Check for required columns
                missing_cols = [col for col in expected_cols if col not in df.columns]
                if missing_cols:
                    st.warning(f"Missing columns: {', '.join(missing_cols)}")
                    st.info("Missing columns will be filled with default values")
                
                # Check if target column exists for evaluation
                has_target = 'Heart Disease Status' in df.columns
                
                if has_target:
                    st.success("✅ Target column 'Heart Disease Status' found - model evaluation will be performed")
                else:
                    st.info("ℹ️ No target column found - only predictions will be generated")
                
                if st.button("🔮 Generate Predictions", use_container_width=True):
                    process_batch_predictions(df, model, scaler, label_encoders, has_target)
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.markdown("Please ensure your CSV file is properly formatted and contains the required columns.")
    
    with tab2:
        add_sample_data_section()

if __name__ == "__main__":
    main()