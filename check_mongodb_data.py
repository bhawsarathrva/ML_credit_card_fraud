"""
Quick script to check MongoDB data structure
"""
from pymongo import MongoClient
import pandas as pd

try:
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print("MongoDB connection successful\n")
    
    # Get database and collection
    db = client["Credit_card"]
    collection = db["Credit"]
    
    # Get document count
    count = collection.count_documents({})
    print(f"Total documents in collection: {count}\n")
    
    if count > 0:
        # Get first document to see structure
        first_doc = collection.find_one()
        
        print("Sample document structure:")
        print("-" * 50)
        for key, value in first_doc.items():
            if key != "_id":
                print(f"{key}: {value} (type: {type(value).__name__})")
        
        print("\n" + "=" * 50)
        print("All column names in dataset:")
        print("=" * 50)
        
        # Get all column names
        df = pd.DataFrame(list(collection.find().limit(100)))
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
        
        print(df.columns.tolist())
        
        print("\n" + "=" * 50)
        print("Checking for target column:")
        print("=" * 50)
        
        if "Class" in df.columns:
            print("[OK] 'Class' column found!")
            print(f"  Unique values: {df['Class'].unique()}")
            print(f"  Value counts:\n{df['Class'].value_counts()}")
        else:
            print("[ERROR] 'Class' column NOT found!")
            print("\nAvailable columns that might be the target:")
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64'] and df[col].nunique() <= 10:
                    print(f"  - {col}: {df[col].unique()}")
    else:
        print("[ERROR] No data in MongoDB collection!")
        print("\nPlease run: python generate_sample_data.py")
    
    client.close()
    
except Exception as e:
    print(f"[ERROR] Error: {str(e)}")
    print("\nMake sure:")
    print("1. MongoDB is running (mongod)")
    print("2. Data is uploaded to MongoDB")
    print("3. Run: python generate_sample_data.py")
