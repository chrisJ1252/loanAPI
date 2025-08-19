import joblib
import pandas as pd 
import logging
from datetime import datetime

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

class ModelWrapper:
    "Wrapper class for ML model"

    def __init__ (self, model_path):
        self.model_info = joblib.load(model_path)
        self.model = self.model_info['model']
        self.feature_names = self.model_info['feature_names']
        self.target_names = self.model_info['target_names']
        self.accuracy = self.model_info['accuracy']
        logger.info(f"Model loaded with accuracy: {self.accuracy:.4f}")

    def predict(self, data):
        "Make predictions and return probabilites"
        predictions = self.model.predict(data)
        probabilities = self.model.predict_proba(data)

        results = []
        for pred, proba in zip(predictions, probabilities):
            result = {
                'prediction_class': self.target_names[pred],
                'predicted_class_id': int(pred),
                'confidence score': float(max(proba)),
                'all_probabilites': {
                    name: float(prob)
                    for name, prob in zip(self.target_names, proba)
                }
            }
            results.append(result)
        return results[0] if len(results) == 1 else results

    def validate_input(self, data):
        "Validate input data"

        self.expected_dtypes = {
            "no_of_dependents": "int64",
            "education": "object",
            "self_employed": "object",
            "income_annum": "int64",
            "loan_ammount": "int64",
            "loan_term": "int64",
            "cibil_score": "int64",
            "residential_assets_value": "int64",
            "commercial_assets_value": "int64",
            "luxury_assets_value": "int64",
            "bank_asset_value ": "int64",
        }
        if isinstance(data, dict):
            data = [data]
        
        df = pd.DataFrame(data)

        missing_features = set(self.feature_names) - set(df.columns)

        if missing_features:
            raise ValueError(f"Missing features: {list(missing_features)}")

        
        for feature, expected_type in self.feature_names:
            if feature not in df.columns:
                raise ValueError(f"Missing feature : {feature}")
            
            actual_series = df[feature]
            if expected_type == 'numeric':
                if not pd.api.types.is_numeric_dtype(actual_series):
                    raise ValueError(f"Feature, {feature} has to be numeric")  
                if actual_series.min() < 0:
                    raise ValueError(f"Feature, {feature} cannot be negative")       
                
            elif expected_type == 'object':
                if not pd.api.types.is_object_dtype(actual_series):
                    raise ValueError(f"Feature, {feature} has to be an object")

