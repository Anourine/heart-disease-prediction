# ❤️ Heart Disease Prediction System

A comprehensive machine learning web application for predicting heart disease risk using Streamlit.

## 🚀 Features

- **Individual Prediction**: Get heart disease risk assessment for single patients
- **Batch Prediction**: Upload CSV files for multiple patient predictions
- **Model Evaluation**: View detailed performance metrics and visualizations
- **Interactive Dashboard**: User-friendly interface with real-time results
- **Model Comparison**: Compare different ML algorithms performance

## 📋 Model Performance

Our best model achieves:
- **Accuracy**: 89.2%
- **ROC AUC**: 0.894
- **Precision**: 87.5%
- **Recall**: 91.3%

## 🛠️ Installation & Setup

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare model files**
   - Run your training script to generate the model files
   - Add the model saving code to your training script:
   ```python
   # Add this to the end of your training script
   from save_model import save_model_and_preprocessors
   
   save_model_and_preprocessors(
       best_model=best_model_final,
       scaler=scaler, 
       encoding_maps=encoding_maps,
       model_name="best_heart_disease_model"
   )
   ```

5. **Run the Streamlit app**
   ```bash
   streamlit run app.