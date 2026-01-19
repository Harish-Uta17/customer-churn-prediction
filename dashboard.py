import streamlit as st
import pandas as pd
import time
import sys
import os
import logging
import plotly.graph_objects as go

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.predictor import ChurnPredictor

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ChurnGuard | AI-Powered Analytics",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADVANCED CUSTOM CSS ---
st.markdown("""
    <style>
    /* Import Modern Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* ============================================ */
    /* SIDEBAR STYLING */
    /* ============================================ */
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2f 0%, #2d2d44 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Default Text Color for Sidebar */
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* --- FIX 1: NUMBER INPUTS (Monthly & Total Charges) --- */
    /* FORCE White Background so the box is clearly visible */
    div[data-testid="stNumberInput"] div[data-baseweb="input"] {
        background-color: #ffffff !important; 
        border: 1px solid #cccccc !important;
        border-radius: 8px !important;
    }

    /* FORCE Black Text so it is readable on the white background */
    div[data-testid="stNumberInput"] input {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        caret-color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* FORCE +/- Buttons to be visible (Light Grey) */
    div[data-testid="stNumberInput"] button {
        background-color: #e0e0e0 !important;
        color: #000000 !important;
        border-color: #cccccc !important;
    }
    
    /* Ensure Icons inside buttons are black */
    div[data-testid="stNumberInput"] button svg {
        fill: #000000 !important;
    }

    /* --- Selectbox Styling (Dark Grey) --- */
    div[data-baseweb="select"] > div {
        background-color: #4b4b6b !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }
    
    /* --- Expander Styling --- */
    div[data-testid="stExpander"] details summary {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 5px !important;
    }

    /* ============================================ */
    /* MAIN DASHBOARD STYLING */
    /* ============================================ */
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 { color: white !important; font-size: 3rem; font-weight: 800; margin: 0; }
    .main-header p { color: rgba(255,255,255,0.9) !important; font-size: 1.1rem; }
    
    /* VIOLET INFO CARDS (Preserved) */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.2);
        color: white !important;
        transition: transform 0.3s ease;
    }
    .info-card:hover { transform: translateY(-5px); }
    .info-card strong { color: rgba(255,255,255,0.8) !important; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; }
    .info-card span { color: #ffffff !important; font-size: 1.5rem; font-weight: 700; display: block; margin-top: 5px; }
    
    /* --- RECOMMENDATIONS (High Contrast) --- */
    .recommendation {
        background-color: #ffffff !important; /* Pure White */
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .recommendation strong { 
        color: #667eea !important; 
        font-weight: 700; 
        font-size: 1.05rem; 
    }
    .recommendation span { 
        color: #000000 !important; /* PURE BLACK TEXT */
        font-weight: 600; 
    }
    
    /* Risk Badges */
    .risk-badge {
        display: inline-block; padding: 8px 16px; border-radius: 20px;
        color: white !important; font-weight: bold; text-align: center;
    }
    .risk-high { background-color: #ff4757; box-shadow: 0 4px 10px rgba(255, 71, 87, 0.3); }
    .risk-medium { background-color: #ffa502; box-shadow: 0 4px 10px rgba(255, 165, 2, 0.3); }
    .risk-low { background-color: #2ed573; box-shadow: 0 4px 10px rgba(46, 213, 115, 0.3); }

    /* Button */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; height: 50px; font-size: 1.1rem;
        font-weight: bold; border-radius: 10px; transition: all 0.3s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }
    
    /* Metrics */
    div[data-testid="stMetricValue"] { color: #2d2d44 !important; }
    div[data-testid="stMetricLabel"] { color: #57606f !important; }

    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        model = ChurnPredictor()
        logger.info("✅ Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None

predictor = load_model()

if predictor is None:
    st.error("❌ Model failed to load. Please check your terminal for errors.")
    st.stop()

# --- HEADER ---
st.markdown("""
    <div class="main-header">
        <h1>🔮 ChurnGuard AI</h1>
        <p>Enterprise Customer Retention Intelligence</p>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR INPUTS ---
st.sidebar.markdown("## 📝 Customer Profile")
st.sidebar.markdown("---")

def user_input_features():
    with st.sidebar.expander("👤 Demographics", expanded=True):
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with st.sidebar.expander("📞 Services", expanded=True):
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = "No"
        if phone == "Yes":
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        # Add-ons
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"

        if internet != "No":
            st.markdown("###### Add-ons")
            online_security = st.selectbox("Online Security", ["Yes", "No"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])

    with st.sidebar.expander("💳 Billing", expanded=True):
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        # Number inputs - Styling handled by CSS above
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0, step=10.0)

    return {
        'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multiple_lines,
        'InternetService': internet, 'OnlineSecurity': online_security,
        'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
        'TechSupport': tech_support, 'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies, 'Contract': contract,
        'PaperlessBilling': paperless, 'PaymentMethod': payment,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }

input_data = user_input_features()

# --- SESSION STATE ---
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# --- LAYOUT ---
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown("### 📊 Overview")
    
    # VIOLET INFO CARDS (Preserved)
    st.markdown(f"""
        <div class="info-card">
            <strong>Contract Type</strong>
            <span>{input_data['Contract']}</span>
        </div>
        <div class="info-card">
            <strong>Tenure</strong>
            <span>{input_data['tenure']} Months</span>
        </div>
        <div class="info-card">
            <strong>Monthly Revenue</strong>
            <span>${input_data['MonthlyCharges']:.2f}</span>
        </div>
        <div class="info-card">
            <strong>Payment Method</strong>
            <span>{input_data['PaymentMethod']}</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 ANALYZE RISK"):
        with st.spinner("Processing..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.005)
                progress.progress(i+1)
            
            try:
                result = predictor.predict_single_customer(input_data)
                st.session_state.prediction = result
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

with col2:
    if st.session_state.prediction:
        result = st.session_state.prediction
        prob = result['probability']
        risk = result['risk_level']
        
        st.markdown("### 🎯 Analysis Results")
        
        # --- FIXED GAUGE CHART (Height Increased) ---
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            title = {'text': "Churn Probability", 'font': {'size': 24, 'color': '#2d2d44'}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "#764ba2"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#e0e0e0",
                'steps': [
                    {'range': [0, 40], 'color': "#2ed573"},
                    {'range': [40, 70], 'color': "#ffa502"},
                    {'range': [70, 100], 'color': "#ff4757"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': prob * 100
                }
            }
        ))
        
        # INCREASED HEIGHT to 400px to prevent cutting off
        fig.update_layout(
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor = "rgba(0,0,0,0)",
            font = {'color': "#2d2d44", 'family': "Inter"},
            height = 400,
            margin = dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Results Row
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", "Churn" if result['prediction'] == 1 else "Retain")
        c2.metric("Probability", f"{prob:.1%}")
        
        risk_class = "risk-high" if risk == "High" else "risk-medium" if risk == "Medium" else "risk-low"
        c3.markdown(f'<div class="risk-badge {risk_class}">{risk} Risk</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 💡 AI Recommendations")
        
        # Recommendations (Dark Text Forced)
        rec_html = ""
        if risk == "High":
             rec_html += '<div class="recommendation">🎁 <strong>Offer:</strong> <span>20% Discount for 6 Months</span></div>'
             rec_html += '<div class="recommendation">📞 <strong>Action:</strong> <span>Schedule Call with Retention Specialist</span></div>'
             rec_html += '<div class="recommendation">🔄 <strong>Switch:</strong> <span>Upgrade to Annual Plan (Waive Fees)</span></div>'
        elif risk == "Medium":
             rec_html += '<div class="recommendation">🛡️ <strong>Add-on:</strong> <span>Free Online Security for 3 Months</span></div>'
             rec_html += '<div class="recommendation">📧 <strong>Email:</strong> <span>Send Customer Satisfaction Survey</span></div>'
        else:
             rec_html += '<div class="recommendation">⭐ <strong>Reward:</strong> <span>Add 500 Loyalty Points</span></div>'
             rec_html += '<div class="recommendation">📺 <strong>Upsell:</strong> <span>Suggest Premium Streaming Bundle</span></div>'
        
        st.markdown(rec_html, unsafe_allow_html=True)
        
    else:
        st.info("👈 Adjust customer profile in the sidebar and click 'Analyze Risk'")