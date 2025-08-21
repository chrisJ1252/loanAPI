# Loan Prediction API Guide

## Base URL
```
loansharks.up.railway.app
```

## Endpoints

### Health Check
**GET** `/health`
```json
{
  "status": "healthy",
  "timestamp": "2025-08-21T12:34:56.789Z",
  "model_loaded": true
}
```

### Model Info
**GET** `/model-info`
```json
{
  "model_type": "Decision Tree",
  "accuracy": 0.97,
  "features": ["feature1", "feature2", ...],
  "classes": ["Approved", "Rejected"],
  "num_featues": 12,
  "num_classes": 2
}
```

### Predict
**POST** `/predict`
**Body Example:**
```json
{
  "no_of_dependents": 1,
  "education": "Graduate",
  "self_employed": "no",
  "income_annum": 120000,
  "loan_ammount": 7000,
  "loan_term": 72,
  "cibil_score": 690,
  "residential_assets_value": 0,
  "commercial_assets_value": 0,
  "luxury_assets_value": 0,
  "bank_asset_value": 0
}
```
**Response Example:**
```json
{
  "status": "success",
  "prediction": { ... },
  "timestamp": "2025-08-21T12:34:56.789Z"
}
```