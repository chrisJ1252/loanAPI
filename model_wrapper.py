import joblib
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelWrapper:    
    def __init__(self, model_path):
        self.model_info = joblib.load(model_path)
        self.model = self.model_info['model']
        self.feature_names = self.model_info['feature_names']
        self.target_names = self.model_info['target_names']
        self.accuracy = self.model_info['accuracy']
        
        if 'preprocessor' in self.model_info:
            self.preprocessor = self.model_info['preprocessor']
            logger.info("Preprocessor loaded from model file")
        else:
            raise ValueError("No preprocessor found. Please provide preprocessor_path or save preprocessor with model.")
        
        logger.info(f"Model loaded with accuracy: {self.accuracy:.4f}")
    
    def preprocess_input(self, data):
        if isinstance(data, dict): # check if the data is an instance of a python dictionary
            data = [data] # need rows not a dictionary, wrap dict in list to create a dataframe with one row
        df = pd.DataFrame(data)
        processed_data = self.preprocessor.transform(df)
        processed_df = pd.DataFrame(processed_data, columns=self.feature_names)
        
        return processed_df
    

    
    def predict(self, data):
        processed_data = self.preprocess_input(data)
        predictions = self.model.predict(processed_data)
        probabilities = self.model.predict_proba(processed_data)
        
        results = []
        for pred, proba in zip(predictions, probabilities):
            result = {
                'prediction_class': self.target_names[pred],
                '\nconfidence_score': float(max(proba)),
                '\nall_probabilities': {
                    name: float(prob)
                    for name, prob in zip(self.target_names, proba)
                }
            }
            results.append(result)
        
        return results[0] if len(results) == 1 else results
    
    def validate_input(self, data):
        expected_features = {
            "no_of_dependents": "numeric",
            "education": "categorical",
            "self_employed": "categorical",
            "income_annum": "numeric",
            "loan_amount": "numeric", 
            "loan_term": "numeric",
            "cibil_score": "numeric",
            "residential_assets_value": "numeric",
            "commercial_assets_value": "numeric",
            "luxury_assets_value": "numeric",
            "bank_asset_value": "numeric",
        }
        
        valid_values = {
            "education": ["Graduate", "Not Graduate"],
            "self_employed": ["Yes", "No"]
        }
        
        if isinstance(data, dict):
            data = [data]
        
        df = pd.DataFrame(data)
        missing_features = set(expected_features.keys()) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {list(missing_features)}")
        
        for feature, expected_type in expected_features.items():
            if feature not in df.columns:
                raise ValueError(f"Missing feature: {feature}")
            
            if expected_type == "numeric":
                if not pd.api.types.is_numeric_dtype(df[feature]) or df[feature].isna().any():
                    raise ValueError(f"Feature '{feature}' must be a number")
                if df[feature].min() < 0:
                    raise ValueError(f"Feature '{feature}' cannot be negative")
            
            elif expected_type == "categorical":
                if feature in valid_values:
                    invalid_values = set(df[feature]) - set(valid_values[feature])
                    if invalid_values:
                        raise ValueError(f"Feature '{feature}' contains invalid values: {invalid_values}. Must be one of: {valid_values[feature]}")
        
        return df