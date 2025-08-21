from flask import Flask, request
from logging import Logger
from model_wrapper import ModelWrapper
from datetime import datetime

app = Flask(__name__)
logger = Logger(__name__)
modelPath = "best_decision_tree.joblib"

try:

    ml_model = ModelWrapper(modelPath)
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    ml_model = None 

@app.route('/')
def home():
    return {
        "service": "Loan Prediction API",
        "\nversion": "0.1.0",
        "\nmodel_accuracy": ml_model.accuracy if ml_model else "N/A",
        "\nendpoints":{
            "predict": "/predict",
            "model_info": "/model-info",
            "health": "/health",
        },
        "\nexample_request":{
            "no_of_dependents": 1,
            "education": "Graduate",
            "self_employed": "no",
            "income_annum": 120000,
            "loan_ammount": 7000,
            "loan_term": 72,
            "cibil_score": 690,
            "residential_assets_value": 0,
            "commercial_assets_value": 0,
            "luxury_assets_value" : 0,
            "bank_asset_value": 0
        }
    }
@app.route('/health')
def health():
    return {
        "\nstatus": "healty" if ml_model else "unhealthy",
        "\ntimestamp": datetime.now().isoformat(),
        "\nmodel_loaded": ml_model is not None
    }
@app.route('/model-info')
def model_info():
    if not ml_model:
        return {"error": "Model not loaded"}, 500
    
    return {
        "\nmodel_type" : "Decision Tree",
        "\naccuracy": ml_model.accuracy,
        "\nfeatures": ml_model.feature_names.tolist(), # needed to be tolist() bc it was ndarray
        "\nclasses": ml_model.target_names.tolist(),
        "\nnum_featues": len(ml_model.feature_names),
        "\nnum_classes": len(ml_model.target_names)

    }

@app.route('/predict', methods = ["POST", "GET"])
def predict():
    if not ml_model:
        return {"error": "Model not loaded"}, 500
    
    try:
        data = request.get_json()
        if not data:
            return {"error": "No JSON data provided"}, 400 
        
        X = ml_model.validate_input(data)

        result = ml_model.predict(X)

        return {
            "status": "success",
            "prediction" : result,
            "timestamp": datetime.now().isoformat()
        }
    except ValueError as v:
        return {"error": str(v)}, 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": "Internal server error"}, 500

@app.errorhandler(404)
def not_found(error):
    return {"error": "Endpoint not found"}, 404

@app.errorhandler(405)
def method_not_allowed(error):
    return {"error": "Method not allowed"}, 405

if(__name__ == '__main__'):
    app.run(debug = True)