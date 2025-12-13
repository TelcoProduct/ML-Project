import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Customer Offer Prediction API", version="1.0")

try:
    model = joblib.load('random_forest_model.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    print("✅ Semua model berhasil dimuat.")
except Exception as e:
    print(f"❌ Error memuat model: {e}")


class CustomerData(BaseModel):
    plan_type: str
    device_brand: str
    avg_data_usage_gb: float
    pct_video_usage: float
    avg_call_duration: float
    sms_freq: int
    monthly_spend: float  
    topup_freq: int
    travel_score: float
    complaint_count: int

    class Config:
        json_schema_extra = {
            "example": {
                 "plan_type": "Postpaid", 
                "device_brand": "Samsung",
                "avg_data_usage_gb": 8.0,    
                "pct_video_usage": 0.4,     
                "avg_call_duration": 2.0,    
                "sms_freq": 2,
                "monthly_spend": 75000,     
                "topup_freq": 1,
                "travel_score": 0.0,
                "complaint_count": 0
            }
        }


@app.post("/predict")
def predict_offer(data: CustomerData):
    try:
        input_df = pd.DataFrame([data.dict()])

        if input_df['avg_data_usage_gb'].iloc[0] == 0:
            input_df['spend_per_gb'] = 0
        else:
            input_df['spend_per_gb'] = input_df['monthly_spend'] / input_df['avg_data_usage_gb']
        processed_data = preprocessor.transform(input_df)


        prediction_index = model.predict(processed_data)[0]
        predicted_label = label_encoder.inverse_transform([prediction_index])[0]


        all_probs = model.predict_proba(processed_data)[0] 
        

        confidence_score = float(np.max(all_probs)) 


        confidence_percent = round(confidence_score * 100, 2)


        return {
            "status": "success",
            "predicted_offer": predicted_label,
            "prediction_code": int(prediction_index),
            "confidence_score": confidence_score,     # Misal: 0.85
            "confidence_percent": f"{confidence_percent}%" # Misal: "85.0%"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoint Cek Kesehatan Server ---
@app.get("/")
def root():
    return {"message": "API Prediksi Penawaran Aktif"}