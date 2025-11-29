import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Business Performance Predictor",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Enterprise Business Performance Predictor")
st.markdown("### Predict business performance based on key metrics")
st.markdown("---")

# Load the model and scaler
@st.cache_resource
def load_model():
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please ensure 'best_model.pkl' and 'scaler.pkl' are in the same directory.")
        return None, None

model, scaler = load_model()

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Performance Metrics")
    employee_productivity = st.slider("Employee Productivity", 0, 100, 75, help="Rate from 0-100")
    customer_satisfaction = st.slider("Customer Satisfaction", 0, 100, 80, help="Rate from 0-100")
    operational_efficiency = st.slider("Operational Efficiency", 0, 100, 70, help="Rate from 0-100")
    revenue_growth = st.slider("Revenue Growth", 0, 100, 65, help="Rate from 0-100")
    cost_efficiency = st.slider("Cost Efficiency", 0, 100, 72, help="Rate from 0-100")

with col2:
    st.subheader("🎯 Strategic Metrics")
    innovation_index = st.slider("Innovation Index", 0, 100, 68, help="Rate from 0-100")
    market_adaptability = st.slider("Market Adaptability", 0, 100, 78, help="Rate from 0-100")
    team_collaboration = st.slider("Team Collaboration", 0, 100, 82, help="Rate from 0-100")
    digital_maturity = st.slider("Digital Maturity", 0, 100, 74, help="Rate from 0-100")
    risk_management = st.slider("Risk Management", 0, 100, 76, help="Rate from 0-100")

st.markdown("---")

# Predict button
if st.button("🔮 Predict Business Performance", type="primary", use_container_width=True):
    if model is not None and scaler is not None:
        # Prepare input data
        input_data = pd.DataFrame({
            'Employee_Productivity': [employee_productivity],
            'Customer_Satisfaction': [customer_satisfaction],
            'Operational_Efficiency': [operational_efficiency],
            'Revenue_Growth': [revenue_growth],
            'Cost_Efficiency': [cost_efficiency],
            'Innovation_Index': [innovation_index],
            'Market_Adaptability': [market_adaptability],
            'Team_Collaboration': [team_collaboration],
            'Digital_Maturity': [digital_maturity],
            'Risk_Management': [risk_management]
        })
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Display result
        st.markdown("---")
        st.markdown("## 🎯 Prediction Result")
        
        # Create three columns for result display
        result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
        
        with result_col2:
            st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>Business Performance: {prediction}</h1>", 
                       unsafe_allow_html=True)
            
            # Performance interpretation
            if prediction >= 1:
                st.success("🌟 Excellent Performance! The business is thriving.")
            # elif prediction >= 60:
            #     st.info("✅ Good Performance! The business is on the right track.")
            # elif prediction >= 40:
            #     st.warning("⚠️ Average Performance. Consider improvement strategies.")
            else:
                st.error("🔴 Low Performance. Immediate action required.")
        
        # Show input summary
        st.markdown("---")
        st.markdown("### 📋 Input Summary")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.write("**Performance Metrics:**")
            st.write(f"- Employee Productivity: {employee_productivity}")
            st.write(f"- Customer Satisfaction: {customer_satisfaction}")
            st.write(f"- Operational Efficiency: {operational_efficiency}")
            st.write(f"- Revenue Growth: {revenue_growth}")
            st.write(f"- Cost Efficiency: {cost_efficiency}")
        
        with summary_col2:
            st.write("**Strategic Metrics:**")
            st.write(f"- Innovation Index: {innovation_index}")
            st.write(f"- Market Adaptability: {market_adaptability}")
            st.write(f"- Team Collaboration: {team_collaboration}")
            st.write(f"- Digital Maturity: {digital_maturity}")
            st.write(f"- Risk Management: {risk_management}")
    else:
        st.error("❌ Cannot make prediction without model files!")

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    """
    This app predicts business performance based on 10 key metrics:
    
    **Performance Metrics:**
    - Employee Productivity
    - Customer Satisfaction
    - Operational Efficiency
    - Revenue Growth
    - Cost Efficiency
    
    **Strategic Metrics:**
    - Innovation Index
    - Market Adaptability
    - Team Collaboration
    - Digital Maturity
    - Risk Management
    
    **Model Accuracy:** 96%
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Information")
st.sidebar.write("**Model Type:** Naive Bayes")
st.sidebar.write("**Training Accuracy:** 94.75%")
st.sidebar.write("**Testing Accuracy:** 96.00%")
st.sidebar.write("**CV Accuracy:** 91.00%")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Enterprise Business Management Analysis Framework | Powered by ML</p>", 
    unsafe_allow_html=True
)