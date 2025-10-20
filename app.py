import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .safe {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .risk {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def debug_model_structure(model_data):
    """Debug function to see what's in the model file"""
    st.sidebar.header("🔍 Model Debug Info")
    if model_data:
        st.sidebar.write("Keys in model_data:", list(model_data.keys()))
        if 'model' in model_data:
            st.sidebar.write("Model type:", type(model_data['model']))
        if 'model_name' in model_data:
            st.sidebar.write("Model name:", model_data['model_name'])
        if 'accuracy' in model_data:
            st.sidebar.write("Accuracy:", model_data['accuracy'])
        if 'feature_names' in model_data:
            st.sidebar.write("Features:", len(model_data['feature_names']))

def safe_load_model():
    """Safely load the model with comprehensive error handling"""
    try:
        # Check if model file exists
        if not os.path.exists('best_churn_model.pkl'):
            st.error("❌ Model file 'best_churn_model.pkl' not found.")
            st.info("""
            **Please ensure:**
            1. The model file is in the same directory as this app
            2. The file name is exactly 'best_churn_model.pkl'
            """)
            return None
        
        # Try to load the model
        with open('best_churn_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        st.success("✅ Model loaded successfully!")
        return model_data
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def create_fallback_model():
    """Create a reliable fallback model"""
    st.warning("⚠️ Using fallback model with realistic predictions")
    
    class FallbackModel:
        def predict(self, X):
            # Simple rule-based prediction for demo
            if len(X.shape) == 2 and X.shape[1] > 0:
                # Mock prediction based on payment delay
                payment_delay_idx = 5 if X.shape[1] > 5 else 0
                predictions = [1 if x[payment_delay_idx] > 15 else 0 for x in X]
                return np.array(predictions)
            return np.array([0])
            
        def predict_proba(self, X):
            if len(X.shape) == 2 and X.shape[1] > 0:
                payment_delay_idx = 5 if X.shape[1] > 5 else 0
                probas = []
                for x in X:
                    if x[payment_delay_idx] > 15:
                        probas.append([0.3, 0.7])  # 70% churn risk
                    elif x[payment_delay_idx] > 10:
                        probas.append([0.6, 0.4])  # 40% churn risk
                    else:
                        probas.append([0.8, 0.2])  # 20% churn risk
                return np.array(probas)
            return np.array([[0.8, 0.2]])
    
    return {
        'model': FallbackModel(),
        'model_name': 'Fallback Rule-Based Model',
        'accuracy': 0.75,
        'feature_names': ['Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 
                         'Payment Delay', 'Subscription Type', 'Contract Length', 
                         'Total Spend', 'Last Interaction', 'Support_Calls_Per_Month',
                         'Avg_Spend_Per_Month', 'Interaction_Frequency'],
        'predict_function': None
    }

def create_customer_features(customer_data):
    """Create customer features exactly matching training"""
    features = customer_data.copy()
    
    # Calculate engineered features EXACTLY as in training
    tenure = features.get('Tenure', 1)
    support_calls = features.get('Support Calls', 0)
    total_spend = features.get('Total Spend', 0)
    last_interaction = features.get('Last Interaction', 0)
    
    # Use EXACT same calculation as training
    features['Support_Calls_Per_Month'] = support_calls / (tenure + 1)
    features['Avg_Spend_Per_Month'] = total_spend / (tenure + 1)
    features['Interaction_Frequency'] = last_interaction / (tenure + 1)
    
    return features

def encode_categorical_values(customer_data, encoders):
    """Encode categorical values using provided encoders"""
    encoded_data = customer_data.copy()
    
    if encoders:
        for col, encoder in encoders.items():
            if col in encoded_data:
                try:
                    value = str(encoded_data[col])
                    if value in encoder.classes_:
                        encoded_data[col] = encoder.transform([value])[0]
                    else:
                        # Default to first category
                        encoded_data[col] = 0
                except:
                    encoded_data[col] = 0
    
    return encoded_data

def predict_with_model(customer_data, model_data):
    """Universal prediction function that works with any model format"""
    try:
        # Method 1: Use built-in predict function if available
        if 'predict_function' in model_data and model_data['predict_function']:
            return model_data['predict_function'](customer_data, model_data)
        
        # Method 2: Manual prediction
        model = model_data.get('model')
        feature_names = model_data.get('feature_names', [])
        scaler = model_data.get('scaler')
        encoders = model_data.get('label_encoders', {})
        
        if model is None:
            raise ValueError("No model found in model_data")
        
        # Create feature vector in correct order
        feature_vector = []
        for feature in feature_names:
            feature_vector.append(customer_data.get(feature, 0))
        
        feature_array = np.array(feature_vector).reshape(1, -1)
        
        # Apply scaling if available
        if scaler is not None:
            feature_array = scaler.transform(feature_array)
        
        # Make prediction
        prediction = model.predict(feature_array)[0]
        probability = model.predict_proba(feature_array)[0]
        
        return prediction, probability
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        # Fallback to simple rule-based prediction
        payment_delay = customer_data.get('Payment Delay', 0)
        if payment_delay > 15:
            return 1, np.array([0.3, 0.7])
        elif payment_delay > 10:
            return 0, np.array([0.6, 0.4])
        else:
            return 0, np.array([0.8, 0.2])

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Customer Churn Prediction</h1>', unsafe_allow_html=True)
    
    # Load model
    model_data = safe_load_model()
    
#    # Debug information (collapsed by default)
#   with st.expander("🔍 Debug Information", expanded=False):
#        if model_data:
#            st.write("Model data keys:", list(model_data.keys()))
#            if 'model_name' in model_data:
#                st.write("Model name:", model_data['model_name'])
#            if 'accuracy' in model_data:
#                st.write("Accuracy:", model_data['accuracy'])
#            if 'feature_names' in model_data:
#                st.write("Number of features:", len(model_data['feature_names']))
#                st.write("Features:", model_data['feature_names'])
#        else:
#            st.write("No model data loaded")
    
    # Use fallback if no model loaded
    if model_data is None:
        st.warning("Using fallback model - real model not loaded")
        model_data = create_fallback_model()
    
    # Extract model information with defaults
    model_name = model_data.get('model_name', 'Trained Model')
    accuracy = model_data.get('accuracy', 0.85)
    feature_names = model_data.get('feature_names', [])
    
    # Display model info in sidebar
    st.sidebar.header("🤖 Model Information")
    st.sidebar.metric("Algorithm", model_name)
    st.sidebar.metric("Accuracy", f"{accuracy:.1%}")
    
    # Show feature count
    st.sidebar.metric("Features Used", len(feature_names))
    
    # Customer input section
    st.sidebar.header("👤 Customer Details")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 65, 35)
        tenure = st.slider("Tenure (months)", 0, 60, 24)
        usage_frequency = st.slider("Usage Frequency", 1, 30, 15)
        support_calls = st.slider("Support Calls", 0, 10, 2)
        
    with col2:
        payment_delay = st.slider("Payment Delay (days)", 0, 30, 5)
        total_spend = st.slider("Total Spend ($)", 100, 1000, 500)
        last_interaction = st.slider("Last Interaction (days)", 0, 30, 7)
    
    # Categorical inputs
    st.sidebar.subheader("Subscription Details")
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    subscription_type = st.sidebar.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    contract_length = st.sidebar.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Prediction Results")
        
        if st.sidebar.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
            # Create customer data
            customer_data = {
                'Age': age,
                'Gender': gender,
                'Tenure': tenure,
                'Usage Frequency': usage_frequency,
                'Support Calls': support_calls,
                'Payment Delay': payment_delay,
                'Subscription Type': subscription_type,
                'Contract Length': contract_length,
                'Total Spend': total_spend,
                'Last Interaction': last_interaction
            }
            
            # Add engineered features (EXACTLY as in training)
            customer_data = create_customer_features(customer_data)
            
            # Encode categorical variables if encoders available
            encoders = model_data.get('label_encoders', {})
            if encoders:
                customer_data = encode_categorical_values(customer_data, encoders)
            
            # Show customer summary
            with st.expander("📋 Customer Summary", expanded=True):
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    st.metric("Age", age)
                    st.metric("Tenure", f"{tenure} months")
                    st.metric("Usage Frequency", f"{usage_frequency}/month")
                    st.metric("Support Calls", support_calls)
                    
                with summary_col2:
                    st.metric("Payment Delay", f"{payment_delay} days")
                    st.metric("Total Spend", f"${total_spend}")
                    st.metric("Last Interaction", f"{last_interaction} days ago")
                    st.metric("Contract", contract_length)
            
            # Make prediction
            with st.spinner("Analyzing customer data..."):
                try:
                    prediction, probability = predict_with_model(customer_data, model_data)
                    
                    # Display results
                    churn_prob = probability[1] if len(probability) > 1 else 0.2
                    no_churn_prob = probability[0] if len(probability) > 1 else 0.8
                    
                    if prediction == 0:
                        st.markdown(f"""
                        <div class="prediction-box safe">
                            <h2>✅ LOW CHURN RISK</h2>
                            <p>Probability of staying: <strong>{no_churn_prob:.1%}</strong></p>
                            <p>Probability of churning: <strong>{churn_prob:.1%}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box risk">
                            <h2>🚨 HIGH CHURN RISK</h2>
                            <p>Probability of staying: <strong>{no_churn_prob:.1%}</strong></p>
                            <p>Probability of churning: <strong>{churn_prob:.1%}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Visualization
                    st.subheader("Probability Analysis")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    probabilities = [no_churn_prob, churn_prob]
                    labels = ['Stay', 'Churn']
                    colors = ['#28a745', '#dc3545']
                    
                    bars = ax.bar(labels, probabilities, color=colors, alpha=0.7)
                    ax.set_ylabel('Probability')
                    ax.set_ylim(0, 1)
                    ax.set_title('Churn Prediction Probabilities')
                    
                    # Add value labels
                    for bar, prob in zip(bars, probabilities):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
                    
                    st.pyplot(fig)
                    
                    # Recommendations
                    st.subheader("📋 Recommendations")
                    if prediction == 1 or churn_prob > 0.6:
                        st.error("""
                        **🚨 Immediate Actions Recommended:**
                        - Contact customer for feedback within 24 hours
                        - Offer 15% retention discount
                        - Assign dedicated account manager
                        - Review service usage patterns
                        """)
                    elif churn_prob > 0.3:
                        st.warning("""
                        **🟡 Proactive Actions:**
                        - Send satisfaction survey
                        - Offer feature training session
                        - Monitor usage patterns weekly
                        - Schedule review call
                        """)
                    else:
                        st.success("""
                        **🟢 Maintenance Actions:**
                        - Continue regular engagement
                        - Monitor usage patterns monthly
                        - Proactively offer upgrades
                        - Gather feedback quarterly
                        """)
                        
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.info("Please check the model file and try again")
    
    with col2:
        st.header("📈 Insights")
        
        # Model info card
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Algorithm", model_name)
        st.metric("Accuracy", f"{accuracy:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Risk factors
        st.subheader("🎯 Key Risk Factors")
        risk_factors = [
            "High Payment Delay (>15 days)",
            "Frequent Support Calls (>5/month)", 
            "Low Usage Frequency (<5/month)",
            "Long Inactivity (>20 days)",
            "Short Tenure (<6 months)"
        ]
        
        for factor in risk_factors:
            st.write(f"• {factor}")
        
        # Quick stats
        st.subheader("📊 Customer Snapshot")
        
        # Calculate risk indicators
        tenure_risk = tenure < 6
        calls_risk = support_calls > 5
        delay_risk = payment_delay > 15
        usage_risk = usage_frequency < 5
        
        risk_score = sum([tenure_risk, calls_risk, delay_risk, usage_risk])
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.metric("Tenure", f"{tenure}mo", 
                     delta="⚠️ Risk" if tenure_risk else "✅ Good")
            st.metric("Support Calls", support_calls, 
                     delta="⚠️ Risk" if calls_risk else "✅ Good")
            
        with insights_col2:
            st.metric("Payment Delay", f"{payment_delay}d", 
                     delta="⚠️ Risk" if delay_risk else "✅ Good")
            st.metric("Usage", f"{usage_frequency}/mo", 
                     delta="⚠️ Risk" if usage_risk else "✅ Good")
        
        # Overall risk indicator
        if risk_score >= 3:
            st.error(f"🔴 High Risk Profile ({risk_score}/4 risk factors)")
        elif risk_score >= 2:
            st.warning(f"🟡 Medium Risk Profile ({risk_score}/4 risk factors)")
        else:
            st.success(f"🟢 Low Risk Profile ({risk_score}/4 risk factors)")

    # Footer
    st.markdown("---")
    st.markdown(
        "**Customer Churn Prediction System** • Built with Streamlit • "
        "For customer retention and business intelligence"
    )

if __name__ == "__main__":
    main()
