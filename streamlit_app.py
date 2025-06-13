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
</style>
""", unsafe_allow_html=True)

# Load the trained model and preprocessing objects
@st.cache_resource
def load_model_and_preprocessors():
    try:
        model = joblib.load('best_heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        return model, scaler, label_encoders
    except FileNotFoundError:
        st.error("Model files not found. Please ensure the model is trained and saved.")
        return None, None, None

# Feature engineering functions
def create_engineered_features(df):
    """Create the same engineered features as in training"""
    # Feature 1: Cholesterol_Ratio
    df['Cholesterol_Ratio'] = df['Cholesterol Level'] / df['Blood Pressure']
    
    # Feature 2: BMI_Category
    bins = [0, 18.5, 24.9, 29.9, np.inf]
    labels = [0, 1, 2, 3]
    df['BMI_Category'] = pd.cut(df['BMI'], bins=bins, labels=labels, right=True, include_lowest=True)
    df['BMI_Category'] = df['BMI_Category'].astype(int)
    
    return df

def preprocess_input_data(input_data, label_encoders):
    """Preprocess input data for prediction"""
    df = pd.DataFrame([input_data])
    
    # Apply label encoding to categorical columns
    categorical_cols = ['Gender', 'Exercise Habits', 'Smoking', 'Family Heart Disease', 
                       'Diabetes', 'High Blood Pressure', 'Low HDL Cholesterol', 
                       'High LDL Cholesterol', 'Alcohol Consumption', 'Stress Level', 
                       'Sugar Consumption',]
    for col in categorical_cols:
        if col in df.columns and col in label_encoders:
            # Handle unseen categories
            le = label_encoders[col]
            df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    
    # Create engineered features
    df = create_engineered_features(df)
    
    return df

# Main app
def main():
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)
    
    # Load model and preprocessors
    model, scaler, label_encoders = load_model_and_preprocessors()
    
    if model is None:
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.selectbox("Choose a page", ["🏠 Home", "🔮 Prediction", "📊 Model Evaluation", "📈 Data Upload & Test"])
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Prediction":
        show_prediction_page(model, scaler, label_encoders)
    elif page == "📊 Model Evaluation":
        show_model_evaluation_page()
    elif page == "📈 Data Upload & Test":
        show_data_upload_page(model, scaler, label_encoders)

def show_home_page():
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

def show_prediction_page(model, scaler, label_encoders):
    st.markdown('<h2 class="sub-header">🔮 Heart Disease Risk Prediction</h2>', unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 100, 45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            BP = st.number_input("Blood Pressure", 80, 200, 120)  
            Cholesterol = st.number_input("Cholesterol Level", 100, 400, 200)
            Exercise = st.number_input("Exercise Habits", [0, 1], format_func=lambda x: "Yes" if x else "No")
            Smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x else "No")
            family_history = st.selectbox("Family Heart Disease	", ["Yes", "No"])
            Diabetes = st.selectbox("Diabetes", ["Yes", "No"])
            BMI = st.selectbox("BMI", 15.0, 50.0, 25.0)
            High Blood Pressure = st.selectbox("High Blood Pressure", ["Yes", "No"])
        
        with col2:
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
            Sleep Hours = st.selectbox("Sleep Hours", ["0-4", "4-8", "8-12", "12+"])
            Sugar Consumption = st.selectbox("Sugar Consumption")
            Triglyceride Level = st.number_input("Triglyceride Level")
            Fasting Blood Sugar = st.slider("Fasting Blood Sugar")
            CRP Level = st.selectbox("CRP Level")
            Homocysteine Level = st.selectbox("Homocysteine Level")
            Cholesterol_Ratio = st.number_input("Cholesterol_Ratio")
            BMI_Category = st.number_input("BMI_Category")
        
        submitted = st.form_submit_button("🔍 Predict Heart Disease Risk")
        
        if submitted:
            # Prepare input data
            input_data = {
                'Age': age,
                'Gender': gender,
                'Blood Pressure': BP,
                'Cholesterol Level': Cholesterol,
                'Exercise Habits': Exercise,
                'Smoking': Smoking,
                'Family Heart Disease': family_history,
                'Diabetes': Diabetes,
                'BMI':BMI,
                'High Blood Pressure': High Blood Pressure,
                'Stress Level': stress_level,
                'Sleep Hours': Sleep Hours,
                'Sugar Consumption': Sugar Consumption,
                'Triglyceride Level': Triglyceride Level,
                'Fasting Blood Sugar': Fasting Blood Sugar,
                'CRP Level': CRP Level,
                'Homocysteine Level': Homocysteine Level,
                'Cholesterol_Ratio': Cholesterol_Ratio,
                'BMI_Category': BMI_Category,
            
            }
            
            # Preprocess and predict
            try:
                processed_data = preprocess_input_data(input_data, label_encoders)
                scaled_data = scaler.transform(processed_data)
                
                prediction = model.predict(scaled_data)[0]
                probability = model.predict_proba(scaled_data)[0]
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    risk_level = "HIGH RISK ⚠️" if prediction == 1 else "LOW RISK ✅"
                    risk_color = "red" if prediction == 1 else "green"
                    st.markdown(f"""
                    <div style="padding: 2rem; background-color: {'#ffebee' if prediction == 1 else '#e8f5e8'}; 
                                border-radius: 10px; text-align: center;">
                        <h2 style="color: {risk_color};">{risk_level}</h2>
                        <h3>Risk Probability: {probability[1]:.2%}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Create probability gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = probability[1] * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Heart Disease Risk (%)"},
                        gauge = {
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
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Risk factors analysis
                st.markdown("### 📋 Risk Factors Analysis")
                risk_factors = []
                if age > 60:
                    risk_factors.append("Age > 60")
                if cholesterol > 240:
                    risk_factors.append("High Cholesterol")
                if blood_pressure > 140:
                    risk_factors.append("High Blood Pressure")
                if diabetes == 1:
                    risk_factors.append("Diabetes")
                if family_history == "Yes":
                    risk_factors.append("Family History")
                if smoking == "Yes":
                    risk_factors.append("Smoking")
                if bmi > 30:
                    risk_factors.append("Obesity (BMI > 30)")
                
                if risk_factors:
                    st.warning("⚠️ Identified Risk Factors:")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.success("✅ No major risk factors identified!")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

def show_model_evaluation_page():
    st.markdown('<h2 class="sub-header">📊 Model Performance Evaluation</h2>', unsafe_allow_html=True)
    
    # Display model metrics (these would be loaded from your training results)
    col1, col2, col3, col4 = st.columns(4)
    
    # Example metrics - replace with actual saved metrics
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
        'Feature': ['Cholesterol Level', 'Age', 'Blood Pressure', 'BMI', 'Heart Rate', 
                   'Stress Level', 'Family History', 'Smoking', 'Exercise Hours', 'Diet'],
        'Importance': [0.15, 0.14, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.07, 0.06]
    }
    
    df_importance = pd.DataFrame(feature_importance)
    
    fig = px.bar(df_importance, x='Importance', y='Feature', orientation='h',
                 title='Top 10 Most Important Features')
    fig.update_traces(marker_color='lightcoral')
    st.plotly_chart(fig, use_container_width=True)

def show_data_upload_page(model, scaler, label_encoders):
    st.markdown('<h2 class="sub-header">📈 Batch Prediction & Model Testing</h2>', unsafe_allow_html=True)
    
    st.markdown("Upload a CSV file containing patient data to get batch predictions.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.markdown("### 📋 Uploaded Data Preview")
            st.dataframe(df.head())
            
            st.markdown(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Check if target column exists for evaluation
            has_target = 'Heart Disease Status' in df.columns
            
            if st.button("🔮 Generate Predictions"):
                with st.spinner("Processing predictions..."):
                    # Prepare features for prediction
                    if has_target:
                        X = df.drop('Heart Disease Status', axis=1)
                        y_true = df['Heart Disease Status']
                    else:
                        X = df
                        y_true = None
                    
                    # Preprocess the data
                    processed_predictions = []
                    
                    for idx, row in X.iterrows():
                        try:
                            processed_data = preprocess_input_data(row.to_dict(), label_encoders)
                            scaled_data = scaler.transform(processed_data)
                            
                            prediction = model.predict(scaled_data)[0]
                            probability = model.predict_proba(scaled_data)[0][1]
                            
                            processed_predictions.append({
                                'Patient_ID': idx + 1,
                                'Prediction': prediction,
                                'Risk_Probability': probability,
                                'Risk_Level': 'High Risk' if prediction == 1 else 'Low Risk'
                            })
                        except Exception as e:
                            st.error(f"Error processing row {idx}: {str(e)}")
                            continue
                    
                    # Create results dataframe
                    results_df = pd.DataFrame(processed_predictions)
                    
                    # Display results
                    st.markdown("### 🎯 Prediction Results")
                    st.dataframe(results_df)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        high_risk_count = len(results_df[results_df['Prediction'] == 1])
                        st.metric("High Risk Patients", high_risk_count)
                    
                    with col2:
                        low_risk_count = len(results_df[results_df['Prediction'] == 0])
                        st.metric("Low Risk Patients", low_risk_count)
                    
                    with col3:
                        avg_risk = results_df['Risk_Probability'].mean()
                        st.metric("Average Risk Probability", f"{avg_risk:.2%}")
                    
                    # Risk distribution chart
                    fig = px.histogram(results_df, x='Risk_Probability', nbins=20,
                                     title='Risk Probability Distribution')
                    fig.update_traces(marker_color='lightblue')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Model evaluation if ground truth is available
                    if has_target and y_true is not None:
                        st.markdown("### 📊 Model Evaluation on Uploaded Data")
                        
                        y_pred = results_df['Prediction'].values
                        y_prob = results_df['Risk_Probability'].values
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_true, y_pred)
                        precision = precision_score(y_true, y_pred)
                        recall = recall_score(y_true, y_pred)
                        f1 = f1_score(y_true, y_pred)
                        roc_auc = roc_auc_score(y_true, y_prob)
                        
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
                        
                        # Confusion matrix
                        cm = confusion_matrix(y_true, y_pred)
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=cm,
                            x=['Predicted Low Risk', 'Predicted High Risk'],
                            y=['Actual Low Risk', 'Actual High Risk'],
                            colorscale='Blues',
                            text=cm,
                            texttemplate="%{text}",
                            textfont={"size": 20}
                        ))
                        
                        fig.update_layout(
                            title='Confusion Matrix',
                            xaxis_title='Predicted',
                            yaxis_title='Actual'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name='heart_disease_predictions.csv',
                        mime='text/csv'
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.markdown("Please ensure your CSV file has the correct format and column names.")

if __name__ == "__main__":
    main()