import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import sys

def generate_sample_data(n_samples=10000, fraud_ratio=0.002):
    """
    Generate synthetic credit card transaction data
    
    Parameters:
    - n_samples: Total number of transactions
    - fraud_ratio: Ratio of fraudulent transactions (default: 0.2%)
    """
    print(f"Generating {n_samples} sample transactions...")
    print(f"Fraud ratio: {fraud_ratio*100:.2f}%")
    
    np.random.seed(42)
    
    # Generate base features
    data = {
        'Time': np.arange(n_samples),
        'Amount': np.abs(np.random.exponential(88, n_samples))  # Transaction amounts
    }
    
    # Generate V1-V28 features (simulating PCA components)
    # These would normally be PCA-transformed features from original data
    for i in range(1, 29):
        if i <= 14:
            # First half: more variation for fraud detection
            data[f'V{i}'] = np.random.randn(n_samples) * (1 + i*0.1)
        else:
            # Second half: less variation
            data[f'V{i}'] = np.random.randn(n_samples) * 0.5
    
    # Generate Class labels (0 = legitimate, 1 = fraud)
    data['Class'] = np.random.choice([0, 1], n_samples, p=[1-fraud_ratio, fraud_ratio])
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Make fraud transactions more distinctive
    fraud_mask = df['Class'] == 1
    if fraud_mask.sum() > 0:
        # Frauds tend to have higher amounts
        df.loc[fraud_mask, 'Amount'] = df.loc[fraud_mask, 'Amount'] * 2.5
        
        # Adjust some V features for frauds to make them more detectable
        for i in [1, 3, 4, 10, 12, 14, 17]:
            df.loc[fraud_mask, f'V{i}'] = df.loc[fraud_mask, f'V{i}'] * 1.8
    
    print(f"\n✓ Generated {len(df)} transactions")
    print(f"  - Legitimate: {sum(df['Class']==0)} ({sum(df['Class']==0)/len(df)*100:.2f}%)")
    print(f"  - Fraud: {sum(df['Class']==1)} ({sum(df['Class']==1)/len(df)*100:.2f}%)")
    
    return df


def upload_to_mongodb(df, mongo_url="mongodb://localhost:27017", 
                     db_name="Credit_card", collection_name="Credit",
                     clear_existing=True):
    """
    Upload DataFrame to MongoDB
    
    Parameters:
    - df: DataFrame to upload
    - mongo_url: MongoDB connection string
    - db_name: Database name
    - collection_name: Collection name
    - clear_existing: Whether to clear existing data
    """
    try:
        print(f"\nConnecting to MongoDB at {mongo_url}...")
        client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
        
        # Test connection
        client.admin.command('ping')
        print("✓ MongoDB connection successful")
        
        # Get database and collection
        db = client[db_name]
        collection = db[collection_name]
        
        # Clear existing data if requested
        if clear_existing:
            existing_count = collection.count_documents({})
            if existing_count > 0:
                print(f"\nClearing {existing_count} existing documents...")
                collection.delete_many({})
                print("✓ Existing data cleared")
        
        # Convert DataFrame to records
        records = df.to_dict('records')
        
        # Insert data
        print(f"\nUploading {len(records)} transactions to MongoDB...")
        result = collection.insert_many(records)
        
        print(f"✓ Successfully uploaded {len(result.inserted_ids)} documents")
        print(f"  Database: {db_name}")
        print(f"  Collection: {collection_name}")
        
        # Verify upload
        final_count = collection.count_documents({})
        fraud_count = collection.count_documents({'Class': 1})
        legitimate_count = collection.count_documents({'Class': 0})
        
        print(f"\n✓ Verification:")
        print(f"  Total documents: {final_count}")
        print(f"  Legitimate: {legitimate_count}")
        print(f"  Fraud: {fraud_count}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"\n✗ Error uploading to MongoDB: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure MongoDB is running: mongod")
        print("2. Check connection string in src/constant.py")
        print("3. Verify network connectivity")
        return False


def save_sample_csv(df, filename="sample_transactions.csv"):
    """
    Save sample data to CSV for testing predictions
    """
    # Create a small sample for prediction testing
    sample_df = df.head(100).copy()
    
    # Remove Class column for prediction input
    if 'Class' in sample_df.columns:
        sample_df = sample_df.drop(columns=['Class'])
    
    sample_df.to_csv(filename, index=False)
    print(f"\n✓ Saved sample prediction file: {filename}")
    print(f"  Use this file to test predictions via /predict endpoint")


def main():
    """
    Main function to generate and upload sample data
    """
    print("="*70)
    print("Credit Card Fraud Detection - Sample Data Generator")
    print("="*70)
    
    # Configuration
    N_SAMPLES = 10000  # Total transactions
    FRAUD_RATIO = 0.002  # 0.2% fraud rate (realistic for credit cards)
    
    MONGO_URL = "mongodb://localhost:27017"
    DB_NAME = "Credit_card"
    COLLECTION_NAME = "Credit"
    
    # Generate data
    df = generate_sample_data(n_samples=N_SAMPLES, fraud_ratio=FRAUD_RATIO)
    
    # Upload to MongoDB
    success = upload_to_mongodb(
        df, 
        mongo_url=MONGO_URL,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        clear_existing=True
    )
    
    if success:
        # Save sample CSV for prediction testing
        save_sample_csv(df, "sample_transactions.csv")
        
        print("\n" + "="*70)
        print("✓ SETUP COMPLETE!")
        print("="*70)
        print("\nNext steps:")
        print("1. Start the Flask app: .\\run_app.bat")
        print("2. Train the model: http://localhost:5000/train")
        print("3. Test predictions: Upload sample_transactions.csv to /predict")
        print("\nFor real fraud detection, replace this data with actual")
        print("credit card transaction data (e.g., from Kaggle dataset)")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("✗ SETUP FAILED")
        print("="*70)
        print("\nPlease fix MongoDB connection and try again")
        sys.exit(1)


if __name__ == "__main__":
    main()
