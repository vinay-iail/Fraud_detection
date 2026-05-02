import pandas as pd
import json
import time
from confluent_kafka import Producer

def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

def main():
    conf = {'bootstrap.servers': 'localhost:9092'}
    producer = Producer(conf)
    topic = 'fraud_transactions'
    
    file_name = '/Users/vinaykumarsimma/Desktop/Capstone_Project/PS_20174392719_1491204439457_log.csv'
    
    print("Starting producer...")
    
    # Read in chunks to not overload memory
    # You can skip rows if you only want to stream test data
    for chunk in pd.read_csv(file_name, chunksize=10000):
        # We only care about transactions where fraud is possible (transfer/cash_out)
        filtered_chunk = chunk[chunk['type'].isin(['TRANSFER', 'CASH_OUT'])]
        
        for index, row in filtered_chunk.iterrows():
            # Convert row to dict
            record = row.to_dict()
            
            # Send to Kafka
            producer.produce(topic, value=json.dumps(record).encode('utf-8'), callback=delivery_report)
            producer.poll(0)
            
            # Simulate real-time delay so it doesn't stream all at once
            time.sleep(0.05)
            
    producer.flush()

if __name__ == '__main__':
    main()
