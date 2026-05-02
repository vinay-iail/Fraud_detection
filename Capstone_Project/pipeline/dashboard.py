import streamlit as st
import sqlite3
import pandas as pd
import time

st.set_page_config(
    page_title="Fraud Detection Live KPI", 
    page_icon="💸",
    layout="wide"
)

# Premium CSS Styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border-left: 5px solid #e91e63;
    }
    .metric-value {
        font-size: 36px;
        font-weight: 800;
        color: #ffffff;
    }
    .metric-label {
        font-size: 14px;
        color: #b0bec5;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💸 Real-time Fraud Detection Dashboard")
st.markdown("Live streaming metrics from Apache Kafka & XGBoost Model")

@st.cache_resource
def get_connection():
    con = sqlite3.connect('predictions.db', check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    return con

try:
    con = get_connection()
except Exception as e:
    st.error(f"Waiting for database initialization... {e}")
    st.stop()

placeholder = st.empty()

while True:
    try:
        # Check if table exists
        table_check = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'", con)
        if table_check.empty:
            with placeholder.container():
                st.warning("⏳ Waiting for Kafka Consumer to write the first transaction...")
            time.sleep(2)
            continue
            
        df = pd.read_sql("SELECT * FROM predictions ORDER BY processed_at DESC LIMIT 100", con)
        
        if len(df) == 0:
            with placeholder.container():
                st.warning("⏳ No data flowing yet. Start the Kafka producer and consumer!")
            time.sleep(2)
            continue
            
        total_tx = pd.read_sql("SELECT COUNT(*) as count FROM predictions", con)['count'][0]
        total_fraud = pd.read_sql("SELECT COUNT(*) as count FROM predictions WHERE isFraud_pred = 1", con)['count'][0]
        
        fraud_rate = (total_fraud / total_tx) * 100 if total_tx > 0 else 0
        
        with placeholder.container():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Transactions Evaluated</div>
                    <div class="metric-value">{total_tx:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #ff9800;">
                    <div class="metric-label">Total Fraud Alerts</div>
                    <div class="metric-value">{total_fraud:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #4caf50;">
                    <div class="metric-label">Current Fraud Rate</div>
                    <div class="metric-value">{fraud_rate:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🚨 Recent Fraud Alerts")
                recent_fraud = df[df['isFraud_pred'] == 1][['amount', 'nameOrig', 'nameDest', 'fraudProbability', 'processed_at']]
                
                if len(recent_fraud) > 0:
                    recent_fraud['amount'] = recent_fraud['amount'].apply(lambda x: '${:,.2f}'.format(x))
                    recent_fraud['fraudProbability'] = recent_fraud['fraudProbability'].apply(lambda x: '{:.2%}'.format(x))
                    st.dataframe(recent_fraud.head(10), use_container_width=True, hide_index=True)
                else:
                    st.success("No recent fraud alerts! ✅")
                    
            with col2:
                st.subheader("📊 Volume by Type")
                type_counts = pd.read_sql("SELECT type, COUNT(*) as count FROM predictions GROUP BY type", con)
                if not type_counts.empty:
                    st.bar_chart(type_counts.set_index('type'), color="#e91e63")
                
    except Exception as e:
        with placeholder.container():
            st.error(f"Error reading database: {e}. Retrying...")
            
    time.sleep(2)
