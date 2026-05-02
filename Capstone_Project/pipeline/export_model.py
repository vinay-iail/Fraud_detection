import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time

def main():
    file_name = '/Users/vinaykumarsimma/Desktop/Capstone_Project/PS_20174392719_1491204439457_log.csv'
    chunks = []
    
    print("Loading data...")
    for chunk in pd.read_csv(file_name, chunksize=500000):
        filtered_chunk = chunk[chunk['type'].isin(['TRANSFER', 'CASH_OUT'])]
        chunks.append(filtered_chunk)
        
    df = pd.concat(chunks, axis=0)
    print(f"Total rows kept: {len(df)}")
    
    # Feature Engineering
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    
    # One-hot encoding
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    # The column will be named 'type_TRANSFER'. Let's rename it to something standard to ensure compatibility
    df.rename(columns={'type_TRANSFER': 'is_transfer'}, inplace=True)
    
    X = df.drop(['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
    y = df['isFraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    ratio = (y == 0).sum() / (y == 1).sum()
    
    xgb_model = XGBClassifier(
        n_estimators=100, # Use slightly fewer trees to save time
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric='logloss'
    )
    
    print("Starting training...")
    start_time = time.time()
    xgb_model.fit(X_train, y_train)
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")
    
    # Save the model features names as well just in case
    print("Feature columns:", X.columns.tolist())
    
    model_path = '/Users/vinaykumarsimma/Desktop/Capstone_Project/pipeline/xgb_model.json'
    xgb_model.save_model(model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()
