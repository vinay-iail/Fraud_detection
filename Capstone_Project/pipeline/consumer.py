import json
import requests
import sqlite3
from confluent_kafka import Consumer, KafkaError

def main():
    # Setup SQLite database connection
    # SQLite perfectly supports dashboard reading while this writes!
    con = sqlite3.connect('predictions.db', check_same_thread=False)
    con.execute('PRAGMA journal_mode=WAL;') # Performance mode for read/write
    
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            step INTEGER,
            type VARCHAR,
            amount DOUBLE,
            nameOrig VARCHAR,
            oldbalanceOrg DOUBLE,
            newbalanceOrig DOUBLE,
            nameDest VARCHAR,
            oldbalanceDest DOUBLE,
            newbalanceDest DOUBLE,
            isFlaggedFraud INTEGER,
            isFraud_pred INTEGER,
            fraudProbability DOUBLE,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.commit()
    
    # Setup Kafka Consumer
    conf = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'fraud-consumer-group',
        'auto.offset.reset': 'earliest'
    }
    
    consumer = Consumer(conf)
    consumer.subscribe(['fraud_transactions'])
    
    print("Consumer started. Listening for transactions...")
    
    API_URL = "http://localhost:8001/predict"
    
    try:
        while True:
            msg = consumer.poll(1.0)
            
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(msg.error())
                    break
            
            # Process incoming Kafka message
            record = json.loads(msg.value().decode('utf-8'))
            
            # Send to FastAPI Model Serving endpoint
            try:
                response = requests.post(API_URL, json=record)
                if response.status_code == 200:
                    pred_data = response.json()
                    
                    # Insert features and prediction into SQLite
                    con.execute("""
                        INSERT INTO predictions (
                            step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
                            nameDest, oldbalanceDest, newbalanceDest, isFlaggedFraud,
                            isFraud_pred, fraudProbability
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        record['step'], record['type'], record['amount'], record['nameOrig'],
                        record['oldbalanceOrg'], record['newbalanceOrig'], record['nameDest'],
                        record['oldbalanceDest'], record['newbalanceDest'], record['isFlaggedFraud'],
                        pred_data['isFraud'], pred_data['fraudProbability']
                    ))
                    con.commit()
                    
                    if pred_data['isFraud'] == 1:
                        print(f"🚨 FRAUD ALERT! Transaction {record['nameOrig']} -> Prob: {pred_data['fraudProbability']:.4f}")
                    else:
                        print(f"✅ Processed transaction {record['nameOrig']} -> Prob: {pred_data['fraudProbability']:.4f}")
                        
                else:
                    print(f"API Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        con.close()

if __name__ == '__main__':
    main()
