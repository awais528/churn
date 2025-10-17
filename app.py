import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the saved model
@st.cache_resource
def load_model():
    with open('best_churn_model.pkl', 'rb') as f:
        return pickle.load(f)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
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
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Load model
    try:
        model_data = load_model()
        model = model_data['model']
        scaler = model_data['scaler']
        encoders = model_data['label_encoders']
        feature_names = model_data['feature_names']
        model_name = model_data['model_name']
        accuracy = model_data['accuracy']
        
        st.sidebar.success(f"✅ Model loaded: {model_name}")
        st.sidebar.metric("Model Accuracy", f"{accuracy:.1%}")
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Sidebar for customer input
    st.sidebar.header("👤 Customer Information")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 65, 35)
        gender = st.selectbox("Gender", ["Female", "Male"])
        tenure = st.slider("Tenure (months)", 0, 60, 24)
        usage_frequency = st.slider("Usage Frequency", 1, 30, 15)
        
    with col2:
        support_calls = st.slider("Support Calls", 0, 10, 2)
        payment_delay = st.slider("Payment Delay (days)", 0, 30, 5)
        subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
        contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
    
    total_spend = st.sidebar.slider("Total Spend ($)", 100, 1000, 500)
    last_interaction = st.sidebar.slider("Last Interaction (days ago)", 0, 30, 7)

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Prediction Results")
        
        if st.sidebar.button("🔮 Predict Churn Risk", type="primary"):
            # Create customer data dictionary
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
            
            # Add engineered features
            customer_data['Support_Calls_Per_Month'] = support_calls / (tenure + 1)
            customer_data['Avg_Spend_Per_Month'] = total_spend / (tenure + 1)
            customer_data['Interaction_Frequency'] = last_interaction / (tenure + 1)
            
            try:
                # Make prediction
                prediction, probability = model_data['predict_function'](customer_data, model_data)
                
                # Display results
                churn_prob = probability[1]
                no_churn_prob = probability[0]
                
                if prediction == 0:
                    st.markdown(f"""
                    <div class="prediction-box safe">
                        <h2>✅ LOW CHURN RISK</h2>
                        <p>This customer is likely to stay with the company.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box risk">
                        <h2>🚨 HIGH CHURN RISK</h2>
                        <p>This customer is at risk of leaving. Immediate action recommended.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability metrics
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric(
                        "Probability of Staying", 
                        f"{no_churn_prob:.1%}",
                        delta=f"{(no_churn_prob-0.5):.1%}" if no_churn_prob > 0.5 else None
                    )
                with metric_col2:
                    st.metric(
                        "Probability of Churning", 
                        f"{churn_prob:.1%}",
                        delta=f"{(churn_prob-0.5):.1%}" if churn_prob > 0.5 else None,
                        delta_color="inverse"
                    )
                
                # Probability chart
                fig, ax = plt.subplots(figsize=(10, 3))
                probabilities = [no_churn_prob, churn_prob]
                labels = ['Stay', 'Churn']
                colors = ['#28a745', '#dc3545']
                
                bars = ax.bar(labels, probabilities, color=colors, alpha=0.7)
                ax.set_ylabel('Probability')
                ax.set_ylim(0, 1)
                ax.set_title('Churn Prediction Probabilities')
                
                # Add value labels on bars
                for bar, prob in zip(bars, probabilities):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
                
                st.pyplot(fig)
                
                # Recommendations based on prediction
                st.subheader("📋 Recommendations")
                if prediction == 1:
                    st.warning("""
                    **Immediate Actions Recommended:**
                    - Contact customer for feedback
                    - Offer special retention discount
                    - Assign dedicated account manager
                    - Review service usage patterns
                    """)
                else:
                    st.success("""
                    **Maintenance Actions:**
                    - Continue regular engagement
                    - Monitor usage patterns
                    - Proactively offer upgrades
                    - Gather feedback for improvement
                    """)
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
    with col2:
        st.header("📈 Model Info")
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Algorithm", model_name)
        st.metric("Accuracy", f"{accuracy:.1%}")
        st.metric("Features Used", len(feature_names))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature importance if available
        if hasattr(model, 'feature_importances_'):
            st.subheader("Top Features")
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True).tail(10)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(feature_importance['feature'], feature_importance['importance'])
            ax.set_xlabel('Importance')
            ax.set_title('Top 10 Feature Importance')
            st.pyplot(fig)
        
        # Quick stats
        st.subheader("Customer Insights")
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.metric("Avg Tenure", f"{tenure} mo")
            st.metric("Support Calls", support_calls)
            
        with insights_col2:
            st.metric("Payment Delay", f"{payment_delay} days")
            st.metric("Usage Freq", f"{usage_frequency}/month")

    # Batch prediction section
    st.header("📁 Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:", batch_data.head())
            
            if st.button("Predict Batch"):
                predictions = []
                probabilities = []
                
                for _, row in batch_data.iterrows():
                    customer_dict = row.to_dict()
                    pred, prob = model_data['predict_function'](customer_dict, model_data)
                    predictions.append(pred)
                    probabilities.append(prob)
                
                batch_data['Churn_Prediction'] = predictions
                batch_data['Churn_Probability'] = [p[1] for p in probabilities]
                
                st.success(f"Predictions completed for {len(batch_data)} customers!")
                st.write("Results:", batch_data[['Churn_Prediction', 'Churn_Probability']].head())
                
                # Download results
                csv = batch_data.to_csv(index=False)
                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Batch prediction error: {e}")

if __name__ == "__main__":
    main()