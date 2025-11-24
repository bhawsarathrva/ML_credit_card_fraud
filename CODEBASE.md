# 📊 Credit Card Fraud Detection - Codebase Analysis

## 🎯 Main File of the Project

**The main file is: `app.py`**

This is the Flask application entry point that serves as the web interface for the entire machine learning project.

---

## 📁 Project Structure Overview

```
credit-card-fraud-detection/
│
├── app.py                          # ⭐ MAIN FILE - Flask web application
├── setup.py                        # Package setup configuration
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker containerization
├── README.md                       # Project documentation
│
├── src/                            # Source code package
│   ├── __init__.py
│   ├── constant.py                 # Project constants
│   ├── exception.py                # Custom exception handling
│   ├── logger.py                   # Logging configuration
│   │
│   ├── components/                 # ML pipeline components
│   │   ├── data_ingestion.py      # Data loading from MongoDB
│   │   ├── data_transformation.py # Feature engineering & preprocessing
│   │   └── model_trainer.py       # Model training & evaluation
│   │
│   ├── pipeline/                  # ML pipelines
│   │   ├── train_pipeline.py      # Training orchestration
│   │   └── predict_pipeline.py    # Prediction orchestration
│   │
│   ├── configuration/             # Configuration management
│   │   └── mongo_db_connection.py # MongoDB connection setup
│   │
│   └── utils/                     # Utility functions
│       └── main_utils.py          # Helper functions
│
├── templates/                     # HTML templates for Flask
│   ├── index.html                 # Home page
│   ├── health.html                # Health check page
│   ├── train.html                 # Training interface
│   └── predict.html               # Prediction interface
│
├── static/                        # Static assets (CSS, JS, images)
├── artifacts/                     # Trained models & preprocessors
├── notebooks/                     # Jupyter notebooks for exploration
├── logs/                          # Application logs
├── config/                        # Configuration files
└── upload_data_to_db/             # Scripts to upload data to MongoDB
```

---

## 🔍 Detailed Analysis of `app.py` (Main File)

### Purpose
`app.py` is the **Flask web application** that provides a user interface and API endpoints for:
1. **Training** machine learning models
2. **Making predictions** on new data
3. **Monitoring** system health and MongoDB connection
4. **Viewing** project status and statistics

### Key Features

#### 1. **MongoDB Integration**
- Connects to MongoDB (Atlas or local) to fetch training data
- Database: Configured via `MONGO_DATABASE_NAME`
- Collection: Configured via `MONGO_COLLECTION_NAME`
- Automatic fallback from Atlas to local MongoDB if connection fails

#### 2. **Web Routes**

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Home dashboard showing project status |
| `/health` | GET | Health check page with MongoDB status |
| `/api/health` | GET | JSON API for health status |
| `/api/status` | GET | JSON API for application status |
| `/train` | GET | Training page & API endpoint |
| `/predict` | GET/POST | Upload CSV for predictions |
| `/mongodb/status` | GET | MongoDB connection details |

#### 3. **Machine Learning Pipelines**

**Training Pipeline** (`/train`):
- Triggers `TraininingPipeline` from `src.pipeline.train_pipeline`
- Orchestrates: Data Ingestion → Transformation → Model Training
- Saves trained model and preprocessor to `artifacts/` folder

**Prediction Pipeline** (`/predict`):
- Accepts CSV file upload
- Loads trained model and preprocessor
- Returns predictions as downloadable CSV

#### 4. **Error Handling**
- Custom exception handling via `CustomException`
- Comprehensive logging using custom logger
- Graceful degradation if MongoDB is unavailable

---

## 🔄 ML Pipeline Flow

### Training Pipeline (`src/pipeline/train_pipeline.py`)

```
1. Data Ingestion (DataIngestion)
   ↓
   - Fetches data from MongoDB
   - Saves raw data locally
   ↓
2. Data Transformation (DataTransformation)
   ↓
   - Feature engineering
   - Preprocessing (scaling, encoding)
   - Train-test split
   ↓
3. Model Training (ModelTrainer)
   ↓
   - Trains ML model (likely XGBoost based on requirements)
   - Evaluates performance
   - Saves model & preprocessor
```

### Prediction Pipeline (`src/pipeline/predict_pipeline.py`)

```
1. Save Input File
   ↓
   - Receives CSV from user
   - Saves to prediction_artifacts/
   ↓
2. Load Models
   ↓
   - Loads trained model from artifacts/model.pkl
   - Loads preprocessor from artifacts/preprocessor.pkl
   ↓
3. Transform & Predict
   ↓
   - Applies same preprocessing
   - Makes predictions
   - Adds prediction column to dataframe
   ↓
4. Return Results
   ↓
   - Saves to predictions/predicted_file.csv
   - Returns file for download
```

---

## 🛠️ Technology Stack

### Backend
- **Flask** - Web framework
- **Python 3.8** - Programming language
- **PyMongo** - MongoDB driver
- **XGBoost** - Machine learning model
- **Pandas** - Data manipulation
- **Scikit-learn** - ML utilities (via imblearn)

### Data Storage
- **MongoDB** - Database for training data (Atlas or local)

### ML Tools
- **MLflow** - Experiment tracking
- **DVC** - Data version control
- **Evidently** - Model monitoring
- **Imbalanced-learn** - Handling class imbalance

### Deployment
- **Docker** - Containerization
- **AWS (boto3)** - Cloud deployment support

---

## 🚀 How to Run the Project

### 1. Setup Environment
```bash
conda create --prefix venv python==3.8 -y
conda activate venv/
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure MongoDB
- Set `MONGO_DB_URL` in `.env` file or environment variables
- Ensure data is uploaded to MongoDB collection

### 4. Run Application
```bash
python app.py
```

The application will start on `http://0.0.0.0:5000`

---

## 📊 Key Components Explained

### 1. **Data Ingestion** (`src/components/data_ingestion.py`)
- Connects to MongoDB
- Fetches credit card transaction data
- Saves to local directory for processing

### 2. **Data Transformation** (`src/components/data_transformation.py`)
- Handles missing values
- Feature scaling/normalization
- Encoding categorical variables
- Handles class imbalance (fraud vs. non-fraud)
- Creates preprocessor pipeline

### 3. **Model Trainer** (`src/components/model_trainer.py`)
- Trains classification model (likely XGBoost)
- Evaluates model performance
- Saves best model to artifacts/

### 4. **Utils** (`src/utils/main_utils.py`)
- Helper functions for:
  - Loading/saving pickle files
  - File operations
  - Common utilities

---

## 🎯 Project Goal

The project aims to:
1. **Detect fraudulent credit card transactions** using machine learning
2. Provide a **web interface** for easy model training and prediction
3. Handle **imbalanced datasets** (fraud cases are rare)
4. Integrate with **MongoDB** for scalable data storage
5. Support **production deployment** via Docker

---

## 📝 Important Notes

### MongoDB Requirement
- The application **requires MongoDB** to be running
- Data must be uploaded to the specified database and collection
- Use scripts in `upload_data_to_db/` to populate MongoDB

### Artifacts
- Trained models are saved in `artifacts/` directory
- Must train model before making predictions
- Artifacts include:
  - `model.pkl` - Trained ML model
  - `preprocessor.pkl` - Data preprocessing pipeline

### Class Imbalance
- Credit card fraud is a highly imbalanced problem
- Project uses `imblearn` library to handle this
- Likely uses techniques like SMOTE or class weighting

---

## 🔗 API Endpoints Summary

### Health & Status
- `GET /health` - HTML health check page
- `GET /api/health` - JSON health status
- `GET /api/status` - Application status
- `GET /mongodb/status` - MongoDB connection details

### ML Operations
- `GET /train` - Training interface (HTML)
- `GET /train` (with Accept: application/json) - Trigger training (API)
- `GET /predict` - Prediction upload page
- `POST /predict` - Submit CSV for predictions

---

## 👨‍💻 Author
**Athrva Bhawsar**
- Email: athrvabh124@gmail.com

---

## 📌 Conclusion

**Main File: `app.py`**

This file is the heart of the application, serving as:
- ✅ Entry point for the web application
- ✅ Orchestrator of ML pipelines
- ✅ API server for training and predictions
- ✅ MongoDB connection manager
- ✅ User interface provider

To understand the complete workflow, start with `app.py` and trace through:
1. Training flow: `app.py` → `train_pipeline.py` → Components
2. Prediction flow: `app.py` → `predict_pipeline.py` → Saved models
