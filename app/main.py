import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("mental_health_api")

# app/ lives one level below the project root
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "Mental_Health_Model.pkl"
FRONTEND_DIR = BASE_DIR / "frontend"
INDEX_PATH = FRONTEND_DIR / "index.html"

TOP_COUNTRIES = {
    "Other",
    "India",
    "USA",
    "Canada",
    "Australia",
    "UK",
    "Germany",
    "Mexico",
    "Turkey",
    "France",
}

# ---------------------------------------------------------------------------
# Feature order — must match the training pipeline exactly
# ---------------------------------------------------------------------------
MODEL_FEATURES = [
    "Study_Hours",
    "Age",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
    "Stress_Level",
    "Gender",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
    "Grouped_country",
]


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class StudentData(BaseModel):
    age: int = Field(..., ge=10, le=100)
    gender: str = Field(..., min_length=1)
    country: str = Field(..., min_length=1)
    academic_level: str = Field(..., min_length=1)
    most_used_platform: str = Field(..., min_length=1)
    purpose_of_use: str = Field(..., min_length=1)
    avg_daily_usage_hours: float = Field(..., ge=0, le=24)
    daily_unlocks: int = Field(..., ge=0)
    study_hours: float = Field(..., ge=0, le=24)
    physical_activity_hours: float = Field(..., ge=0, le=24)
    sleep_hours_per_night: float = Field(..., ge=0, le=24)
    stress_level: str = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    predicted_mental_health_score: float


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    logger.info("Loading model from %s …", MODEL_PATH)
    pipeline = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully (%s)", type(pipeline).__name__)
    return pipeline


model = load_model()

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Mental Health Signal API",
    version="1.0.0",
    description="Predict student mental-health scores from daily habits.",
)

# CORS — allow the frontend (and dev tools) to reach the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static assets (CSS, JS)
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def home():
    """Serve the single-page frontend."""
    return FileResponse(INDEX_PATH)


@app.get("/health")
def health():
    """Quick liveness / readiness check."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: StudentData):
    """Run the trained pipeline on user-supplied data and return a score."""
    try:
        country_group = data.country if data.country in TOP_COUNTRIES else "Other"

        # Build a single-row DataFrame with only the 12 features the model expects.
        # Note: "Country" is NOT a model feature — we use "Grouped_country" instead.
        input_row = pd.DataFrame(
            [
                {
                    "Age": data.age,
                    "Gender": data.gender,
                    "Academic_Level": data.academic_level,
                    "Most_Used_Platform": data.most_used_platform,
                    "Purpose_Of_Use": data.purpose_of_use,
                    "Avg_Daily_Usage_Hours": data.avg_daily_usage_hours,
                    "Daily_Unlocks": data.daily_unlocks,
                    "Study_Hours": data.study_hours,
                    "Physical_Activity_Hours": data.physical_activity_hours,
                    "Sleep_Hours_Per_Night": data.sleep_hours_per_night,
                    "Stress_Level": data.stress_level,
                    "Grouped_country": country_group,
                }
            ]
        )

        # Ensure column order matches what the model was trained on
        input_row = input_row[MODEL_FEATURES]

        prediction = model.predict(input_row)[0]
        score = round(float(prediction), 2)
        logger.info("Prediction: %.2f (country_group=%s)", score, country_group)
        return PredictionResponse(predicted_mental_health_score=score)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal server error") from exc

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)