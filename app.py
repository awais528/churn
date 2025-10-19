import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

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

# Load the saved model with robust error handling
@st.cache_resource
def load_model():
    try:
        with open('best_churn_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        st.sidebar.success("✅ Model loaded successfully!")
        return model_data
    except FileNotFoundError:
        st.sidebar.error("❌ Model file not found. Please run the training script first.")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        return None

def create_customer_features(customer_data):
    """Create customer features with engineered features"""
    features = customer_data.copy()
    
    # Calculate engineered features
    tenure = features.get('Tenure', 1)
    support_calls = features.get('Support Calls', 0)
    total_spend = features.get('Total Spend', 0)
    last_interaction = features.get('Last Interaction', 0)
    
    features['Support_Calls_Per_Month'] = support_calls / (tenure + 1)
    features['Avg_Spend_Per_Month'] = total_spend / (tenure + 1)
    features['Interaction_Frequency'] = last_interaction / (tenure + 1)
    
    return features

def safe_encode_value(value, encoder, feature_name):
    """Safely encode categorical values with fallback"""
    try:
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        else:
            st.warning(f"Unknown {feature_name}: '{value}'. Using default.")
            return encoder.transform([encoder.classes_[0]])[0]
    except Exception as e:
        st.error(f"Encoding error for {feature_name}: {e}")
        return 0

def predict_single_customer(customer_data, model_data):
    """Make prediction for a single customer with robust error handling"""
    try:
        if 'predict_function' in model_data:
            # Use the model's built-in prediction function
            return model_data['predict_function'](customer_data, model_data)
        else:
            # Fallback prediction method
            return fallback_prediction(customer_data, model_data)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0, np.array([0.7, 0.3])  # Safe default

def fallback_prediction(customer_data, model_data):
    """Fallback prediction method"""
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    # Create feature vector
    feature_vector = []
    for feature in feature_names:
        if feature in customer_data:
            feature_vector.append(customer_data[feature])
        else:
            # Use mean or default value
            default_value = model_data.get('feature_means', {}).get(feature, 0)
            feature_vector.append(default_value)
    
    feature_array = np.array(feature_vector).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(feature_array)[0]
    probability = model.predict_proba(feature_array)[0]
    
    return prediction, probability

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Load model
    model_data = load_model()
    
    if model_data is None:
        st.error("""
        ## Model not found!
        
        Please make sure you have:
        1. Run the training script to generate `best_churn_model.pkl`
        2. The model file is in the same directory as this app
        3. All dependencies are installed
        
        Check the requirements.txt file for required packages.
        """)
        return

    # Display model info
    model_name = model_data.get('model_name', 'Unknown')
    accuracy = model_data.get('accuracy', 0)
    
    st.sidebar.header("🤖 Model Information")
    st.sidebar.metric("Algorithm", model_name)
    st.sidebar.metric("Accuracy", f"{accuracy:.1%}")
    st.sidebar.metric("Features", len(model_data.get('feature_names', [])))
    
    # Display categorical mappings for reference
    if 'categorical_mappings' in model_data:
        with st.sidebar.expander("📋 Category Mappings"):
            for col, mapping in model_data['categorical_mappings'].items():
                st.write(f"**{col}:**")
                st.write(mapping['mapping'])

    # Customer input section
    st.sidebar.header("👤 Customer Information")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 65, 35)
        tenure = st.slider("Tenure (months)", 0, 60, 24)
        usage_frequency = st.slider("Usage Frequency", 1, 30, 15)
        support_calls = st.slider("Support Calls", 0, 10, 2)
        
    with col2:
        payment_delay = st.slider("Payment Delay (days)", 0, 30, 5)
        total_spend = st.slider("Total Spend ($)", 100, 1000, 500)
        last_interaction = st.slider("Last Interaction (days ago)", 0, 30, 7)
    
    # Categorical inputs with validation
    st.sidebar.subheader("Subscription Details")
    
    # Get available categories from model
    gender_options = ["Female", "Male"]
    subscription_options = ["Basic", "Standard", "Premium"]
    contract_options = ["Monthly", "Quarterly", "Annual"]
    
    # Update options based on model training data if available
    if 'categorical_mappings' in model_data:
        if 'Gender' in model_data['categorical_mappings']:
            gender_options = model_data['categorical_mappings']['Gender']['classes']
        if 'Subscription Type' in model_data['categorical_mappings']:
            subscription_options = model_data['categorical_mappings']['Subscription Type']['classes']
        if 'Contract Length' in model_data['categorical_mappings']:
            contract_options = model_data['categorical_mappings']['Contract Length']['classes']
    
    gender = st.sidebar.selectbox("Gender", gender_options)
    subscription_type = st.sidebar.selectbox("Subscription Type", subscription_options)
    contract_length = st.sidebar.selectbox("Contract Length", contract_options)

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Prediction Results")
        
        if st.sidebar.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
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
            customer_data = create_customer_features(customer_data)
            
            # Show customer summary
            with st.expander("📋 Customer Summary", expanded=True):
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.metric("Age", age)
                    st.metric("Tenure", f"{tenure} months")
                    st.metric("Usage", f"{usage_frequency}/month")
                
                with summary_col2:
                    st.metric("Support Calls", support_calls)
                    st.metric("Payment Delay", f"{payment_delay} days")
                    st.metric("Total Spend", f"${total_spend}")
                
                with summary_col3:
                    st.metric("Last Contact", f"{last_interaction} days ago")
                    st.metric("Subscription", subscription_type)
                    st.metric("Contract", contract_length)
            
            # Make prediction
            with st.spinner("Analyzing customer data..."):
                prediction, probability = predict_single_customer(customer_data, model_data)
            
            # Display results
            churn_prob = probability[1]
            no_churn_prob = probability[0]
            
            # Results display
            if prediction == 0:
                st.markdown(f"""
                <div class="prediction-box safe">
                    <h2>✅ LOW CHURN RISK</h2>
                    <p>This customer has a <strong>{no_churn_prob:.1%}</strong> probability of staying.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box risk">
                    <h2>🚨 HIGH CHURN RISK</h2>
                    <p>This customer has a <strong>{churn_prob:.1%}</strong> probability of churning.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability metrics
            st.subheader("Probability Analysis")
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                delta_stay = no_churn_prob - 0.5
                st.metric(
                    "Probability of Staying", 
                    f"{no_churn_prob:.1%}",
                    delta=f"{delta_stay:+.1%}" if abs(delta_stay) > 0.01 else None,
                    delta_color="normal" if delta_stay > 0 else "inverse"
                )
            
            with metric_col2:
                delta_churn = churn_prob - 0.5
                st.metric(
                    "Probability of Churning", 
                    f"{churn_prob:.1%}",
                    delta=f"{delta_churn:+.1%}" if abs(delta_churn) > 0.01 else None,
                    delta_color="inverse"
                )
            
            # Probability visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Bar chart
            probabilities = [no_churn_prob, churn_prob]
            labels = ['Stay', 'Churn']
            colors = ['#28a745', '#dc3545']
            
            bars = ax1.bar(labels, probabilities, color=colors, alpha=0.7)
            ax1.set_ylabel('Probability')
            ax1.set_ylim(0, 1)
            ax1.set_title('Churn Prediction Probabilities')
            
            # Add value labels on bars
            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
            
            # Pie chart
            ax2.pie(probabilities, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Probability Distribution')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Risk level analysis
            st.subheader("📊 Risk Level Analysis")
            if churn_prob < 0.3:
                risk_level = "🟢 LOW RISK"
                risk_color = "green"
            elif churn_prob < 0.7:
                risk_level = "🟡 MEDIUM RISK"
                risk_color = "orange"
            else:
                risk_level = "🔴 HIGH RISK"
                risk_color = "red"
            
            st.markdown(f"**Risk Level:** <span style='color:{risk_color}; font-weight:bold;'>{risk_level}</span>", unsafe_allow_html=True)
            
            # Recommendations
            st.subheader("📋 Action Recommendations")
            if prediction == 1 or churn_prob > 0.6:
                st.warning("""
                **🚨 IMMEDIATE ACTIONS RECOMMENDED:**
                
                🔹 **Contact customer** within 24 hours for feedback
                🔹 **Offer retention discount** (15-20%)
                🔹 **Assign dedicated account manager**
                🔹 **Review service usage patterns**
                🔹 **Schedule follow-up call** in 1 week
                
                **Priority:** High
                """)
            elif churn_prob > 0.3:
                st.info("""
                **🟡 PROACTIVE ACTIONS RECOMMENDED:**
                
                🔸 **Send satisfaction survey**
                🔸 **Offer feature training session**
                🔸 **Check for usage issues**
                🔸 **Monitor payment patterns**
                🔸 **Schedule quarterly review**
                
                **Priority:** Medium
                """)
            else:
                st.success("""
                **🟢 MAINTENANCE ACTIONS:**
                
                ✅ **Continue regular engagement**
                ✅ **Monitor usage patterns**
                ✅ **Proactively offer upgrades**
                ✅ **Gather feedback for improvement**
                ✅ **Maintain current service level**
                
                **Priority:** Low
                """)
    
    with col2:
        st.header("📈 Model Insights")
        
        # Model performance card
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Algorithm", model_name)
        st.metric("Accuracy", f"{accuracy:.1%}")
        st.metric("Features Used", len(model_data.get('feature_names', [])))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Key risk factors
        st.subheader("🎯 Key Risk Factors")
        risk_factors = [
            ("High Payment Delay", "> 15 days"),
            ("Frequent Support Calls", "> 5 calls/month"),
            ("Low Usage Frequency", "< 5 times/month"),
            ("Long Inactivity", "> 20 days"),
            ("Short Tenure", "< 6 months")
        ]
        
        for factor, threshold in risk_factors:
            st.write(f"• **{factor}** ({threshold})")
        
        # Quick stats
        st.subheader("📊 Customer Snapshot")
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.metric("Tenure", f"{tenure} mo", delta="Good" if tenure > 12 else "Watch")
            st.metric("Support Calls", support_calls, delta="High" if support_calls > 5 else "Normal")
            
        with insights_col2:
            st.metric("Payment Delay", f"{payment_delay}d", delta="Risk" if payment_delay > 15 else "OK")
            st.metric("Usage", f"{usage_frequency}/mo", delta="Low" if usage_frequency < 8 else "Good")

    # Batch prediction section
    st.header("📁 Batch Prediction")
    
    st.markdown("""
    <div class="warning-box">
    💡 <strong>Batch Prediction Note:</strong> Upload a CSV file with customer data. 
    The file should contain columns matching the training features.
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write("**Uploaded Data Preview:**", batch_data.head())
            
            if st.button("🔮 Predict Batch", type="secondary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                predictions = []
                probabilities = []
                
                total_rows = len(batch_data)
                for i, (_, row) in enumerate(batch_data.iterrows()):
                    customer_dict = row.to_dict()
                    customer_dict = create_customer_features(customer_dict)
                    
                    pred, prob = predict_single_customer(customer_dict, model_data)
                    predictions.append(pred)
                    probabilities.append(prob[1])  # Churn probability
                    
                    # Update progress
                    progress = (i + 1) / total_rows
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {i+1}/{total_rows} customers...")
                
                # Add predictions to data
                batch_data['Churn_Prediction'] = predictions
                batch_data['Churn_Probability'] = probabilities
                batch_data['Risk_Level'] = batch_data['Churn_Probability'].apply(
                    lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.3 else 'Low'
                )
                
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"✅ Predictions completed for {len(batch_data)} customers!")
                
                # Show summary
                churn_count = sum(predictions)
                st.write(f"**Summary:** {churn_count} customers predicted to churn ({churn_count/len(predictions):.1%})")
                
                # Show results
                st.write("**Prediction Results:**")
                result_cols = ['Churn_Prediction', 'Churn_Probability', 'Risk_Level']
                st.dataframe(batch_data[result_cols].head(10))
                
                # Download results
                csv = batch_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Batch prediction error: {e}")

    # Footer
    st.markdown("---")
    st.markdown(
        "**Customer Churn Prediction System** • Built with Streamlit • "
        "For customer retention and business intelligence"
    )

if __name__ == "__main__":
    main()
