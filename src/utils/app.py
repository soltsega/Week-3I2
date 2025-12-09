from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from predict import PremiumOptimizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Insurance Premium Optimizer API",
    description="API for predicting optimal insurance premiums",
    version="1.0.0"
)

# Initialize the predictor
predictor = PremiumOptimizer()

class PredictionInput(BaseModel):
    SumInsured: float
    CalculatedPremiumPerTerm: float
    TotalPremium: float
    TotalClaims: float
    ExcessSelected_freq: float
    CoverCategory_freq: float
    CoverType_freq: float
    CoverGroup_freq: float
    Section_freq: float

@app.get("/")
async def root():
    return {"message": "Insurance Premium Optimizer API is running"}

@app.post("/predict", response_model=Dict[str, Any])
async def predict(input_data: PredictionInput):
    try:
        result = predictor.predict(input_data.dict())
        if result.get('status') == 'error':
            raise HTTPException(status_code=400, detail=result.get('message'))
        return result
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)