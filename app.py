from flask import Flask, render_template, jsonify, request, send_file
from src.exception import CustomException
from src.logger import logging as lg
import os
import sys
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from src.pipeline.train_pipeline import TraininingPipeline
from src.pipeline.predict_pipeline import PredictionPipeline
from src.constant import MONGO_DATABASE_NAME, MONGO_COLLECTION_NAME, MONGO_DB_URL
from src.configuration.mongo_db_connection import MongoDBClient

app = Flask(__name__)
mongo_client = None
mongo_db = None
mongo_collection = None


def initialize_mongodb_connection():
    global mongo_client, mongo_db, mongo_collection
    
    try:
        lg.info("Initializing MongoDB connection...")
        
        if not os.getenv("MONGO_DB_URL"):
            os.environ["MONGO_DB_URL"] = MONGO_DB_URL
            lg.info("Set MONGO_DB_URL from constants")
        
        try:
            mongo_db_client = MongoDBClient(database_name=MONGO_DATABASE_NAME)
            mongo_client = mongo_db_client.client
            mongo_db = mongo_db_client.database
            lg.info(f"Connected to MongoDB Atlas - Database: {MONGO_DATABASE_NAME}")
        except Exception as atlas_error:
            lg.warning(f"Failed to connect via MongoDBClient (Atlas): {atlas_error}")
            lg.info("Attempting local MongoDB connection...")
            mongo_client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
            mongo_db = mongo_client[MONGO_DATABASE_NAME]
            lg.info(f"Connected to local MongoDB - Database: {MONGO_DATABASE_NAME}")
        
        mongo_client.admin.command('ping')
        lg.info("MongoDB connection verified successfully")
        
        mongo_collection = mongo_db[MONGO_COLLECTION_NAME]
        
        collection_count = mongo_collection.count_documents({})
        lg.info(f"Collection '{MONGO_COLLECTION_NAME}' found with {collection_count} documents")
        
        if collection_count == 0:
            lg.warning(f"Collection '{MONGO_COLLECTION_NAME}' is empty. Please upload data to MongoDB.")
        
        return True
        
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        lg.error(f"MongoDB connection failed: {e}")
        lg.error("Please ensure MongoDB is running and accessible.")
        return False
    except Exception as e:
        lg.error(f"Error initializing MongoDB connection: {e}")
        raise CustomException(e, sys)


@app.route("/")
def home():
    try:
        document_count = 0
        mongodb_connected = mongo_client is not None and mongo_db is not None
        
        if mongo_collection:
            try:
                document_count = mongo_collection.count_documents({})
            except:
                pass
        
        return render_template('index.html', 
                             database_name=MONGO_DATABASE_NAME,
                             collection_name=MONGO_COLLECTION_NAME,
                             mongodb_connected=mongodb_connected,
                             document_count=document_count)
    except Exception as e:
        lg.error(f"Error rendering home page: {e}")
        return render_template('index.html', 
                             database_name=MONGO_DATABASE_NAME,
                             collection_name=MONGO_COLLECTION_NAME,
                             mongodb_connected=False,
                             document_count=0)


@app.route("/api/status")
def api_status():
    try:
        status = {
            "status": "running",
            "mongodb_connected": mongo_client is not None and mongo_db is not None,
            "database": MONGO_DATABASE_NAME,
            "collection": MONGO_COLLECTION_NAME
        }
        
        if mongo_collection:
            status["document_count"] = mongo_collection.count_documents({})
        
        return jsonify(status)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/health")
def health_check():
    """
    Health check endpoint to verify MongoDB connection
    """
    try:
        if mongo_client is None or mongo_db is None:
            return render_template('health.html',
                                 status="unhealthy",
                                 mongodb="not connected",
                                 database=MONGO_DATABASE_NAME,
                                 collection=MONGO_COLLECTION_NAME,
                                 document_count=0,
                                 error="MongoDB not initialized")

        mongo_client.admin.command('ping')
        
        collection_count = mongo_collection.count_documents({}) if mongo_collection else 0
        
        return render_template('health.html',
                             status="healthy",
                             mongodb="connected",
                             database=MONGO_DATABASE_NAME,
                             collection=MONGO_COLLECTION_NAME,
                             document_count=collection_count,
                             error=None)
    except Exception as e:
        lg.error(f"Health check failed: {e}")
        return render_template('health.html',
                             status="unhealthy",
                             mongodb="not connected",
                             database=MONGO_DATABASE_NAME,
                             collection=MONGO_COLLECTION_NAME,
                             document_count=0,
                             error=str(e))


@app.route("/api/health")
def api_health():
    try:
        if mongo_client is None or mongo_db is None:
            return jsonify({
                "status": "unhealthy",
                "mongodb": "not connected"
            }), 503

        mongo_client.admin.command('ping')
        
        collection_count = mongo_collection.count_documents({}) if mongo_collection else 0
        
        return jsonify({
            "status": "healthy",
            "mongodb": "connected",
            "database": MONGO_DATABASE_NAME,
            "collection": MONGO_COLLECTION_NAME,
            "document_count": collection_count
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503


@app.route("/train", methods=['GET'])
def train_route():
    wants_json = request.headers.get('Accept', '').find('application/json') != -1
    
    if not wants_json:
        return render_template('train.html',
                             database_name=MONGO_DATABASE_NAME,
                             collection_name=MONGO_COLLECTION_NAME)
    
    try:
        lg.info("Starting training pipeline...")
        
        if mongo_client is None or mongo_db is None:
            lg.error("MongoDB not connected. Reinitializing...")
            if not initialize_mongodb_connection():
                return jsonify({
                    "status": "error",
                    "message": "MongoDB connection failed. Please check your MongoDB setup."
                }), 500
        
        train_pipeline = TraininingPipeline()
        train_pipeline.run_pipeline()
        
        lg.info("Training pipeline completed successfully")
        return jsonify({
            "status": "success",
            "message": "Training Completed Successfully"
        })
        
    except Exception as e:
        lg.error(f"Training pipeline failed: {e}")
        error_message = str(e)
        return jsonify({
            "status": "error",
            "message": f"Training failed: {error_message}"
        }), 500


@app.route('/predict', methods=['POST', 'GET'])
def upload():
    """
    Prediction route - GET shows upload page, POST processes file
    """
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                return jsonify({
                    "status": "error",
                    "message": "No file provided"
                }), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    "status": "error",
                    "message": "No file selected"
                }), 400
            
            if not file.filename.endswith('.csv'):
                return jsonify({
                    "status": "error",
                    "message": "Only CSV files are supported"
                }), 400
            
            lg.info("Starting prediction pipeline...")
            
            prediction_pipeline = PredictionPipeline(request)
            prediction_file_detail = prediction_pipeline.run_pipeline()

            lg.info("Prediction completed. Downloading prediction file.")
            return send_file(
                prediction_file_detail.prediction_file_path,
                download_name=prediction_file_detail.prediction_file_name,
                as_attachment=True
            )
        else:
            # Render prediction/upload page
            return render_template('predict.html',
                                 database_name=MONGO_DATABASE_NAME,
                                 collection_name=MONGO_COLLECTION_NAME)
            
    except Exception as e:
        lg.error(f"Prediction pipeline failed: {e}")
        error_message = str(e)
        if request.method == 'POST':
            return jsonify({
                "status": "error",
                "message": f"Prediction failed: {error_message}"
            }), 500
        else:
            return render_template('predict.html',
                                 database_name=MONGO_DATABASE_NAME,
                                 collection_name=MONGO_COLLECTION_NAME,
                                 error=error_message)


@app.route("/mongodb/status")
def mongodb_status():
    try:
        if mongo_client is None or mongo_db is None:
            return jsonify({
                "connected": False,
                "message": "MongoDB not initialized"
            }), 503
        
        mongo_client.admin.command('ping')
        
        collection_count = mongo_collection.count_documents({}) if mongo_collection else 0
        
        db_stats = mongo_db.command("dbstats")
        
        return jsonify({
            "connected": True,
            "database": MONGO_DATABASE_NAME,
            "collection": MONGO_COLLECTION_NAME,
            "document_count": collection_count,
            "database_size": db_stats.get("dataSize", 0),
            "collections": db_stats.get("collections", 0)
        })
        
    except Exception as e:
        return jsonify({
            "connected": False,
            "error": str(e)
        }), 503


if __name__ == "__main__":
    lg.info("Starting Flask application...")
    if initialize_mongodb_connection():
        lg.info("MongoDB connection established. Starting Flask server...")
    else:
        lg.warning("MongoDB connection failed. Application will start but training may fail.")
    
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)