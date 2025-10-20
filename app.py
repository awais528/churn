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

def safe_load_model():
    """Safely load the model with comprehensive error handling"""
    try:
        # Check if model file exists
        if not os.path.exists('best_churn_model.pkl'):
            st.error("❌ Model file 'best_churn_model.pkl' not found.")
            st.info("""
            **To fix this:**
            1. Run the training script locally to generate the model file
            2. Make sure 'best_churn_model.pkl' is in your GitHub repository
            3. The file should be in the same directory as this app
            """)
            return None
        
        # Try to load the model
        with open('best_churn_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        st.sidebar.success("✅ Model loaded successfully!")
        return model_data
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("""
        **Common solutions:**
        - The model file might be corrupted
        - There might be version compatibility issues
        - Try regenerating the model file with updated dependencies
        """)
        return None

def create_sample_model():
    """Create a simple fallback model for demonstration"""
    st.warning("⚠️ Using demonstration mode with sample data")
    
    # Create a simple mock model
    class MockModel:
        def predict(self, X):
            return np.array([0])  # Always predict no churn for demo
            
        def predict_proba(self, X):
            return np.array([[0.7, 0.3]])  # 70% no churn, 30% churn
    
    return {
        'model': MockModel(),
        'model_name': 'Demo Model',
        'accuracy': 0.75,
        'feature_names': ['Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 
                         'Payment Delay', 'Subscription Type', 'Contract Length', 
                         'Total Spend', 'Last Interaction'],
        'predict_function': lambda customer_data, model_data: (0, np.array([0.7, 0.3]))
    }

def create_customer_features(customer_data, encoders):
    """Create customer features that match the training exactly"""
    features = customer_data.copy()
    
    # Calculate engineered features EXACTLY as in training
    try:
        tenure = features.get('Tenure', 1)
        support_calls = features.get('Support Calls', 0)
        total_spend = features.get('Total Spend', 0)
        last_interaction = features.get('Last Interaction', 0)
        
        # Use EXACT same calculation as training
        features['Support_Calls_Per_Month'] = support_calls / (tenure + 1)
        features['Avg_Spend_Per_Month'] = total_spend / (tenure + 1)
        features['Interaction_Frequency'] = last_interaction / (tenure + 1)
    except Exception as e:
        st.error(f"Error in feature engineering: {e}")
        # Set defaults if calculation fails
        features['Support_Calls_Per_Month'] = 0
        features['Avg_Spend_Per_Month'] = 0
        features['Interaction_Frequency'] = 0
    
    return features

def encode_categorical_features(customer_data, encoders):
    """Encode categorical features using the same encoders from training"""
    encoded_data = customer_data.copy()
    
    for feature, encoder in encoders.items():
        if feature in encoded_data:
            try:
                value_str = str(encoded_data[feature])
                if value_str in encoder.classes_:
                    encoded_data[feature] = encoder.transform([value_str])[0]
                else:
                    # Use first class as default for unknown values
                    encoded_data[feature] = encoder.transform([encoder.classes_[0]])[0]
            except Exception as e:
                st.error(f"Error encoding {feature}: {e}")
                encoded_data[feature] = 0
    
    return encoded_data

def predict_churn(customer_data, model_data):
    """Make prediction using the model's predict_function"""
    try:
        if 'predict_function' in model_data:
            return model_data['predict_function'](customer_data, model_data)
        else:
            # Fallback to manual prediction
            model = model_data['model']
            scaler = model_data.get('scaler', None)
            feature_names = model_data.get('feature_names', [])
            
            # Create feature vector in correct order
            feature_vector = []
            for feature in feature_names:
                feature_vector.append(customer_data.get(feature, 0))
            
            feature_array = np.array(feature_vector).reshape(1, -1)
            
            # Apply scaling if needed
            if scaler is not None:
                feature_array = scaler.transform(feature_array)
            
            prediction = model.predict(feature_array)[0]
            probability = model.predict_proba(feature_array)[0]
            
            return prediction, probability
            
    except Exception as e:
        st.error(f"Prediction error: {e}")
        # Return safe default
        return 0, np.array([0.8, 0.2])

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Customer Churn Prediction</h1>', unsafe_allow_html=True)
    
    # Load model or use demo
    model_data = safe_load_model()
    
    if model_data is None:
        st.warning("Running in demonstration mode with sample predictions")
        model_data = create_sample_model()
    
    # Display model info - FIXED: Properly extract model information
    model_name = model_data.get('model_name', 'Unknown Model')
    accuracy = model_data.get('accuracy', 0)
    feature_names = model_data.get('feature_names', [])
    encoders = model_data.get('label_encoders', {})
    
    st.sidebar.header("🤖 Model Information")
    st.sidebar.metric("Algorithm", model_name)
    st.sidebar.metric("Accuracy", f"{accuracy:.1%}")
    
    # Show feature info in expander
    with st.sidebar.expander("📋 Model Features"):
        st.write(f"Number of features: {len(feature_names)}")
        for feature in feature_names:
            st.write(f"• {feature}")
    
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
                    # Use the model's built-in predict function
                    prediction, probability = predict_churn(customer_data, model_data)
                    
                    # Display results
                    churn_prob = probability[1]
                    no_churn_prob = probability[0]
                    
                    if prediction == 0:
                        st.markdown(f"""
                        <div class="prediction-box safe">
                            <h2>✅ LOW CHURN RISK</h2>
                            <p>Probability of staying: <strong>{no_churn_prob:.1%}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box risk">
                            <h2>🚨 HIGH CHURN RISK</h2>
                            <p>Probability of churning: <strong>{churn_prob:.1%}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Probability metrics
                    st.subheader("Probability Analysis")
                    metric_col1, metric_col2 = st.columns(2)
                    
                    with metric_col1:
                        st.metric("Probability of Staying", f"{no_churn_prob:.1%}")
                        
                    with metric_col2:
                        st.metric("Probability of Churning", f"{churn_prob:.1%}")
                    
                    # Visualization
                    fig, ax = plt.subplots(figsize=(10, 4))
                    
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
                        st.warning("""
                        **🚨 Immediate Actions Recommended:**
                        - Contact customer for feedback
                        - Offer retention discount
                        - Assign dedicated account manager
                        - Review service usage patterns
                        """)
                    elif churn_prob > 0.3:
                        st.info("""
                        **🟡 Proactive Actions:**
                        - Send satisfaction survey
                        - Offer feature training
                        - Monitor usage patterns
                        - Schedule review call
                        """)
                    else:
                        st.success("""
                        **🟢 Maintenance Actions:**
                        - Continue regular engagement
                        - Monitor usage patterns
                        - Proactively offer upgrades
                        - Gather feedback
                        """)
                        
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    st.info("Try using the demo model or retraining the model")
    
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
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            risk_tenure = "⚠️" if tenure < 6 else "✅"
            risk_calls = "⚠️" if support_calls > 5 else "✅"
            st.metric("Tenure", f"{tenure}mo", delta=risk_tenure)
            st.metric("Support Calls", support_calls, delta=risk_calls)
            
        with insights_col2:
            risk_delay = "⚠️" if payment_delay > 15 else "✅"
            risk_usage = "⚠️" if usage_frequency < 5 else "✅"
            st.metric("Payment Delay", f"{payment_delay}d", delta=risk_delay)
            st.metric("Usage", f"{usage_frequency}/mo", delta=risk_usage)

    # Demo notice
    if model_data.get('model_name') == 'Demo Model':
        st.markdown("""
        <div class="warning-box">
        💡 <strong>Demo Mode:</strong> This is running with sample predictions. 
        To use the real model, make sure 'best_churn_model.pkl' is available.
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "**Customer Churn Prediction System** • Built with Streamlit • "
        "For customer retention and business intelligence"
    )

if __name__ == "__main__":
    main()
