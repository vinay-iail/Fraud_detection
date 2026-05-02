# Capstone Project: ML Pipeline Deployment Guide
**Team 3 - Real-Time Fraud Detection Pipeline**

This guide provides the exact step-by-step commands to flawlessly run your machine learning pipeline both locally and publicly on the AWS cloud.

---

## 🛑 Phase 0: Resetting Environments to Zero
Before recording a demo, you must reset the historical database so your Streamlit dashboard starts at exactly zero transactions.

### 1. Wipe the AWS Cloud Database
Log into AWS, navigate to the pipeline, and safely delete the database:
```bash
ssh -i /Users/vinaykumarsimma/Desktop/Capstone_Project/capstone-key.pem ubuntu@54.159.126.135
cd Capstone_Project/pipeline
pkill -f 'uvicorn' || true
pkill -f 'streamlit' || true
pkill -f 'python3 consumer.py' || true
pkill -f 'python3 producer.py' || true
rm -f predictions.db
exit
```

### 2. Wipe the Local Mac Database
Run this in a local Mac terminal to reset your local dashboard:
```bash
cd /Users/vinaykumarsimma/Desktop/Capstone_Project/pipeline
pkill -f 'uvicorn' || true
pkill -f 'streamlit' || true
pkill -f 'python3 consumer.py' || true
pkill -f 'python3 producer.py' || true
rm -f predictions.db
```

---

## 🚀 Phase 1: Local Mac Deployment
Use this method if you want to visually show all terminal logs flying by in different windows.

**1. Open 4 separate Terminal tabs on your Mac.**
Ensure you are in the correct folder for every single tab:
```bash
cd /Users/vinaykumarsimma/Desktop/Capstone_Project/pipeline
```

**2. In Tab 1 (Boot the Kafka Cluster):**
```bash
docker-compose up -d
```

**3. In Tab 2 (Start the Backend FastAPI Model):**
```bash
uvicorn api:app --reload --port 8001
```

**4. In Tab 3 (Start the Kafka Consumer):**
```bash
python3 consumer.py
```

**5. In Tab 4 (Start the Dashboard):**
```bash
streamlit run dashboard.py
```
*(Keep the browser window open and verify it says 0 transactions).*

**6. Back in Tab 1 (Ignite the Live Stream!):**
```bash
python3 producer.py
```

---

## 🌍 Phase 2: Public AWS Cloud Deployment
Use this method for a professional, single-window deployment using background processes (`nohup`).

**1. Log into your Cloud Server:**
```bash
ssh -i /Users/vinaykumarsimma/Desktop/Capstone_Project/capstone-key.pem ubuntu@54.159.126.135
```

**2. Navigate to your project folder:**
```bash
cd Capstone_Project/pipeline
```

**3. Reactivate your Python ML Environment:**
```bash
source venv/bin/activate
```

**4. Start the Apache Kafka Cluster:**
```bash
sudo docker-compose up -d
```

**5. Start the FastAPI XGBoost Model:**
```bash
nohup uvicorn api:app --host 0.0.0.0 --port 8001 &
```

**6. Start the Kafka Consumer (The Receiver):**
```bash
nohup python3 consumer.py &
```

**7. Start the Streamlit Dashboard (The UI):**
```bash
nohup python3 -m streamlit run dashboard.py --server.port 8502 --server.address 0.0.0.0 &
```

**8. THE MAGIC COMMAND (Starts the live data stream!):**
```bash
nohup python3 producer.py &
```

*(Immediately open Google Chrome and navigate to `http://54.159.126.135:8502` to see your public cloud dashboard!)*
