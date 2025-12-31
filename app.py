from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables
MODEL_PATH = 'artifacts/fraud_model.pkl'
SCALER_PATH = 'artifacts/scaler.pkl'
DATASET_PATH = 'notebooks/creditCardFraud_Data.csv'

# Feature names (all columns except target)
FEATURE_NAMES = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 
                 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 
                 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 
                 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

def train_model():
    """Train the fraud detection model from CSV"""
    print("=" * 80)
    print("TRAINING FRAUD DETECTION MODEL")
    print("=" * 80)
    
    # Load dataset
    print(f"\n1. Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"   Dataset shape: {df.shape}")
    
    # Prepare features and target
    X = df.drop('default payment next month', axis=1)
    y = df['default payment next month']
    
    print(f"\n2. Class distribution:")
    print(f"   Non-Fraud (0): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.2f}%)")
    print(f"   Fraud (1): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.2f}%)")
    
    # Train-test split
    print(f"\n3. Splitting data (80-20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature scaling
    print(f"\n4. Scaling features...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE
    print(f"\n5. Applying SMOTE for class balancing...")
    print(f"   Before SMOTE - Fraud: {(y_train == 1).sum()}, Non-Fraud: {(y_train == 0).sum()}")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    print(f"   After SMOTE - Fraud: {(y_train_smote == 1).sum()}, Non-Fraud: {(y_train_smote == 0).sum()}")
    
    # Train XGBoost model
    print(f"\n6. Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_train_smote, y_train_smote)
    
    # Evaluate model
    print(f"\n7. Evaluating model on test set...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"   ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    
    # Save model and scaler
    print(f"\n8. Saving model and scaler...")
    os.makedirs('artifacts', exist_ok=True)
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"   Model saved to: {MODEL_PATH}")
    
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   Scaler saved to: {SCALER_PATH}")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}\n")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

def load_model():
    """Load trained model and scaler"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

@app.route('/')
def home():
    """Home page with prediction form"""
    model_trained = os.path.exists(MODEL_PATH)
    return render_template('simple_predict.html', 
                         model_trained=model_trained,
                         feature_names=FEATURE_NAMES)

@app.route('/train', methods=['POST'])
def train():
    """Train the model"""
    try:
        metrics = train_model()
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully!',
            'metrics': {
                'accuracy': f"{metrics['accuracy']*100:.2f}%",
                'precision': f"{metrics['precision']*100:.2f}%",
                'recall': f"{metrics['recall']*100:.2f}%",
                'f1_score': f"{metrics['f1_score']*100:.2f}%",
                'roc_auc': f"{metrics['roc_auc']*100:.2f}%"
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Training failed: {str(e)}'
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction on custom input"""
    try:
        # Load model and scaler
        model, scaler = load_model()
        
        if model is None or scaler is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not trained yet. Please train the model first.'
            }), 400
        
        # Get input data from form
        input_data = []
        for feature in FEATURE_NAMES:
            value = request.form.get(feature)
            if value is None or value == '':
                return jsonify({
                    'status': 'error',
                    'message': f'Missing value for {feature}'
                }), 400
            input_data.append(float(value))
        
        # Convert to numpy array and reshape
        input_array = np.array(input_data).reshape(1, -1)
        
        # Scale the input
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Prepare result
        result = {
            'status': 'success',
            'prediction': int(prediction),
            'prediction_label': 'FRAUD' if prediction == 1 else 'NON-FRAUD',
            'confidence': {
                'non_fraud': f"{prediction_proba[0]*100:.2f}%",
                'fraud': f"{prediction_proba[1]*100:.2f}%"
            },
            'risk_level': 'HIGH RISK' if prediction == 1 else 'LOW RISK'
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/status')
def status():
    """Check if model is trained"""
    model_trained = os.path.exists(MODEL_PATH)
    return jsonify({
        'model_trained': model_trained,
        'model_path': MODEL_PATH if model_trained else None,
        'dataset_path': DATASET_PATH
    })

if __name__ == '__main__':
    print("\n" + "="*80)
    print("CREDIT CARD FRAUD DETECTION - FLASK APP")
    print("="*80)
    print(f"\nDataset: {DATASET_PATH}")
    print(f"Model will be saved to: {MODEL_PATH}")
    print(f"\nStarting Flask server on http://localhost:5000")
    print("="*80 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)