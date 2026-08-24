from pathlib import Path
from typing import Optional
import json
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
MODEL = joblib.load(BASE / 'hotel_star_model.joblib')
META = json.loads((BASE / 'model_metadata.json').read_text(encoding='utf-8'))
FEATURES = META['features']
CITY_FREQ = META['city_frequency']

app = FastAPI(title='Hotel Star Rating API', version='1.0.0')

class HotelFeatures(BaseModel):
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    mmt_holidayiq_review_count: Optional[float] = None
    mmt_review_count: Optional[float] = None
    mmt_review_score: Optional[float] = None
    mmt_tripadvisor_count: Optional[float] = None
    location_rating: Optional[float] = Field(default=None, ge=0, le=5)
    amenity_count: Optional[float] = None
    room_type_count: Optional[float] = None
    room_feature_count: Optional[float] = None
    overview_word_count: Optional[float] = None
    property_type: str = 'Hotel'
    is_value_plus: str = 'no'

def to_model_row(x: HotelFeatures) -> pd.DataFrame:
    lat, lon = x.latitude, x.longitude
    if lat is not None and lon is not None and not (6 <= lat <= 38 and 68 <= lon <= 98):
        lat, lon = np.nan, np.nan
    city_frequency = CITY_FREQ.get(x.city, 1) if x.city else 1
    row = {
        'latitude_clean': lat,
        'longitude_clean': lon,
        'mmt_holidayiq_review_count': x.mmt_holidayiq_review_count,
        'mmt_review_count': x.mmt_review_count,
        'mmt_review_score': x.mmt_review_score,
        'mmt_tripadvisor_count': x.mmt_tripadvisor_count,
        'location_rating_clean': x.location_rating,
        'amenity_count': x.amenity_count,
        'room_type_count': x.room_type_count,
        'room_feature_count': x.room_feature_count,
        'overview_word_count': x.overview_word_count,
        'city_frequency': city_frequency,
        'property_type': x.property_type,
        'is_value_plus': x.is_value_plus.lower(),
    }
    return pd.DataFrame([row], columns=FEATURES)

@app.get('/')
def root():
    return {'name':'Hotel Star Rating API','version':'1.0.0','docs':'/docs'}

@app.get('/health')
def health():
    return {'status':'healthy','model_loaded':True}

@app.post('/predict')
def predict(x: HotelFeatures):
    try:
        frame = to_model_row(x)
        pred = int(MODEL.predict(frame)[0])
        proba = MODEL.predict_proba(frame)[0]
        classes = [int(c) for c in MODEL.classes_]
        return {
            'predicted_star_rating': pred,
            'probabilities': {str(c): round(float(p), 6) for c,p in zip(classes,proba)}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
