# 💳 Credit Card Fraud Detection System
## End-to-End ML Project with SMOTE-Tomek & MongoDB

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.0-green)](https://flask.palletsprojects.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Compatible-brightgreen)](https://www.mongodb.com/)
[![SMOTE](https://img.shields.io/badge/Technique-SMOTE--Tomek-orange)](https://imbalanced-learn.org/)
[![License](https://img.shields.io/badge/License-Educational-yellow)](LICENSE)

---

## 🎯 Project Overview

A **production-ready Credit Card Fraud Detection System** that uses advanced machine learning techniques to identify fraudulent transactions in highly imbalanced datasets.

### ✨ Key Features

- 🔍 **Advanced Fraud Detection** - SMOTE-Tomek & KNN-SMOTE comparison
- 🗄️ **MongoDB Integration** - Scalable data storage and retrieval
- 🤖 **Multiple ML Models** - Evaluates 7 different classifiers automatically
- 📊 **Comprehensive Metrics** - Precision, Recall, F1-Score, ROC-AUC
- 🌐 **Web Interface** - User-friendly Flask application with Visualizations
- 📈 **Detailed Reporting** - HTML reports, ROC curves, Confusion Matrices
- 🚀 **Production Ready** - Error handling, logging, and documentation

---

## 🏗️ Architecture

```
MongoDB → Data Ingestion → Preprocessing → SMOTE-Tomek → Model Training → Prediction
   ↓            ↓              ↓              ↓              ↓              ↓
Storage    Fetch Data    RobustScaler   Balance Data   7 Models      Fraud/Legit
```

### Pipeline Components:

1. **Data Ingestion** - Fetches transaction data from MongoDB
2. **Data Transformation** - Applies RobustScaler and SMOTE-Tomek balancing
3. **Model Training** - Trains and evaluates 7 ML models
4. **Model Selection** - Chooses best model based on F1 score
5. **Prediction** - Classifies new transactions as fraud (1) or legitimate (0)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or 3.12
- MongoDB (local or Atlas)
- Virtual environment

### Installation

```bash
# 1. Clone or download the project
cd credit-card-fraud-detection

# 2. Activate virtual environment
.\venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate sample data (for testing)
python generate_sample_data.py

# 5. Start the application
.\run_app.bat  # Windows
```

### Access the Application

Open your browser and navigate to: **http://localhost:5000**

---

## 📊 SMOTE-Tomek Technique

### The Challenge: Severe Class Imbalance

Credit card fraud datasets are highly imbalanced:
- **Legitimate transactions**: ~99.8%
- **Fraudulent transactions**: ~0.2%

Traditional ML models fail because they predict everything as legitimate!

### The Solution: SMOTE-Tomek

**SMOTE-Tomek** combines two powerful techniques:

1. **SMOTE (Synthetic Minority Over-sampling)**
   - Generates synthetic fraud examples
   - Uses K-Nearest Neighbors algorithm
   - Increases minority class representation

2. **Tomek Links Removal**
   - Removes noisy borderline samples
   - Cleans decision boundaries
   - Improves model generalization

### Results

| Metric | Before SMOTE | After SMOTE-Tomek |
|--------|--------------|-------------------|
| Training Fraud Ratio | 0.2% | 33% |
| Model Recall | 60% | 90%+ |
| Frauds Caught | 6 out of 10 | 9 out of 10 |

---

## 🤖 Machine Learning Models

The system evaluates **7 different classifiers**:

1. **Logistic Regression** - Fast baseline with class balancing
2. **K-Nearest Neighbors** - Distance-based classification
3. **Decision Tree** - Interpretable tree-based model
4. **Random Forest** - Ensemble of decision trees
5. **Gradient Boosting** - Sequential boosting algorithm
6. **XGBoost** - Optimized gradient boosting
7. **AdaBoost** - Adaptive boosting ensemble

### Model Selection Criteria

Models are ranked by:
1. **F1 Score** (Primary) - Balance between precision and recall
2. **Recall** (Secondary) - Fraud detection rate ≥ 75%
3. **Precision** (Tertiary) - Fraud prediction accuracy
4. **ROC-AUC** (Overall) - Discrimination ability

---

## 📈 Evaluation Metrics

### Fraud-Specific Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Recall** | % of frauds caught | ≥ 75% |
| **Precision** | % of fraud alerts that are real | ≥ 70% |
| **F1 Score** | Balance between precision & recall | ≥ 70% |
| **ROC-AUC** | Overall discrimination ability | ≥ 85% |

### Confusion Matrix

```
                Predicted
             Legit    Fraud
Actual Legit   TN       FP     ← False alarms
       Fraud   FN       TP     ← Frauds caught
                ↑        ↑
           Missed   Detected
```

- **TP (True Positive)**: Frauds correctly detected ✅
- **TN (True Negative)**: Legitimate correctly identified ✅
- **FP (False Positive)**: Legitimate flagged as fraud ⚠️
- **FN (False Negative)**: Frauds missed ❌ CRITICAL

---

## 🌐 Web Interface

### Available Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Home dashboard with system status |
| `/health` | GET | Health check page |
| `/train` | GET | Model training interface |
| `/predict` | GET/POST | Upload CSV for predictions |
| `/api/health` | GET | JSON health status |
| `/api/status` | GET | JSON application status |
| `/mongodb/status` | GET | MongoDB connection details |

### Usage Examples

#### Train Model
```bash
# Via web interface
http://localhost:5000/train

# Via API
curl -H "Accept: application/json" http://localhost:5000/train
```

#### Make Predictions
```bash
# Upload CSV via web interface
http://localhost:5000/predict

# Or use API
curl -X POST -F "file=@transactions.csv" http://localhost:5000/predict
```

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── app.py                          # Flask application (main entry)
├── run_app.bat                     # Quick start script
├── generate_sample_data.py         # Sample data generator
├── requirements_fixed.txt          # Python dependencies
│
├── src/                            # Source code
│   ├── constant.py                 # Configuration
│   ├── exception.py                # Error handling
│   ├── logger.py                   # Logging
│   │
│   ├── components/                 # ML components
│   │   ├── data_ingestion.py      # MongoDB data fetching
│   │   ├── data_transformation.py # SMOTE-Tomek preprocessing
│   │   └── model_trainer.py       # Model training & evaluation
│   │
│   ├── pipeline/                   # ML pipelines
│   │   ├── train_pipeline.py      # Training orchestration
│   │   └── predict_pipeline.py    # Prediction orchestration
│   │
│   └── utils/                      # Utilities
│       └── main_utils.py           # Helper functions
│
├── artifacts/                      # Generated files
│   ├── Card_data.csv              # Raw data
│   ├── model.pkl                  # Trained model
│   ├── preprocessor.pkl           # Preprocessing pipeline
│   ├── smote.pkl                  # SMOTE object
│   └── model_report.txt           # Performance report
│
├── templates/                      # HTML templates
│   ├── index.html                 # Home page
│   ├── health.html                # Health check
│   ├── train.html                 # Training UI
│   └── predict.html               # Prediction UI
│
└── docs/                          # Documentation
    ├── PROJECT_DOCUMENTATION.md   # Complete guide
    ├── QUICK_START.md             # Quick start guide
    ├── CHANGES_SUMMARY.md         # What changed
    └── ISSUE_RESOLUTION.md        # Troubleshooting
```

---

## 🔧 Configuration

### MongoDB Settings (`src/constant.py`)

```python
MONGO_DATABASE_NAME = "Credit_card"
MONGO_COLLECTION_NAME = "Credit"
MONGO_DB_URL = "mongodb://localhost:27017"
TARGET_COLUMN = "Class"  # 0 = Legitimate, 1 = Fraud
```

### SMOTE-Tomek Settings (`src/components/data_transformation.py`)

```python
SMOTETomek(
    sampling_strategy=0.5,  # Fraud becomes 50% of legitimate
    random_state=42,
    n_jobs=-1
)
```

### Model Selection Thresholds (`src/components/model_trainer.py`)

```python
expected_recall = 0.75      # Minimum 75% fraud detection
expected_f1_score = 0.70    # Minimum 70% F1 score
```

---

## 📊 Dataset Requirements

### Expected Format

Your MongoDB collection should contain:

**Features** (30 columns):
- `Time` - Seconds elapsed since first transaction
- `V1` to `V28` - PCA-transformed anonymized features
- `Amount` - Transaction amount
- `Class` - 0 (Legitimate) or 1 (Fraud)

### Example Document

```json
{
  "Time": 0,
  "V1": -1.3598071336738,
  "V2": -0.0727811733098497,
  "V3": 2.53634673796914,
  ...
  "V28": -0.0210530534538215,
  "Amount": 149.62,
  "Class": 0
}
```

### Sample Data

Use the included script to generate sample data:

```bash
python generate_sample_data.py
```

This creates 10,000 synthetic transactions with realistic fraud patterns.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) | Complete system documentation |
| [QUICK_START.md](QUICK_START.md) | 5-minute setup guide |
| [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) | Summary of all changes |
| [ISSUE_RESOLUTION.md](ISSUE_RESOLUTION.md) | Troubleshooting guide |

---

## 🐛 Troubleshooting

### MongoDB Connection Failed

```bash
# Start MongoDB
mongod

# Or check connection string
# Update src/constant.py with correct URL
```

### Module Not Found

```bash
# Install dependencies
pip install -r requirements_fixed.txt
```

### Low Model Performance

- Ensure sufficient fraud examples in training data
- Adjust SMOTE sampling_strategy
- Try different model hyperparameters
- Check data quality

---

## 📈 Expected Performance

Based on typical credit card fraud datasets:

| Metric | Expected Range |
|--------|---------------|
| Accuracy | 95% - 99% |
| Precision | 70% - 90% |
| **Recall** | **75% - 95%** ⭐ |
| **F1 Score** | **75% - 92%** ⭐ |
| ROC-AUC | 90% - 99% |

**Note**: Recall and F1 Score are prioritized over accuracy for fraud detection.

---

## 🎓 Key Learnings

1. **Imbalanced Data** - SMOTE-Tomek effectively handles severe class imbalance
2. **Metric Selection** - F1 Score and Recall matter more than accuracy
3. **Preprocessing** - RobustScaler handles outliers better than StandardScaler
4. **Model Comparison** - Ensemble methods typically perform best
5. **Business Impact** - Minimize false negatives (missed frauds)

---

## 🚀 Future Enhancements

- [ ] Real-time prediction API with FastAPI
- [ ] Deep learning models (LSTM, Autoencoders)
- [ ] Anomaly detection techniques
- [ ] Model monitoring dashboard
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure)
- [ ] CI/CD pipeline

---

## 👨‍💻 Author

**Athrva Bhawsar**
- Email: athrvabh124@gmail.com
- GitHub: [bhawsarathrva](https://github.com/bhawsarathrva)

---

## 📄 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

- **Dataset**: Kaggle Credit Card Fraud Detection
- **SMOTE**: Chawla et al. (2002)
- **Tomek Links**: Tomek (1976)
- **Libraries**: scikit-learn, imbalanced-learn, XGBoost

---

## 📞 Support

For issues, questions, or contributions:

1. Check [QUICK_START.md](QUICK_START.md) for setup help
2. Review [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) for details
3. See [ISSUE_RESOLUTION.md](ISSUE_RESOLUTION.md) for troubleshooting

---

## ⭐ Star This Project

If you find this project helpful, please consider giving it a star!

---

**Built with ❤️ for fraud detection and machine learning education**

**Happy Fraud Detection! 🔍💳🚀**
