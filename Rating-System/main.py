# main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import pandas as pd
import numpy as np

# ========================
# Load model at startup
# ========================

# This is the pipeline you saved from Colab (preprocess + XGBoost)
MODEL_PATH = "rating_model.joblib"
model = joblib.load(MODEL_PATH)

# These MUST match what the pipeline expects (same as training)
TEXT_COL = "review_text"

NUMERIC_COLS = [
    "avg_rating_total_reviews_written",
    "age",
    "rating_rest",
    "review_count",
    "popularity_score",
    "avg_price",
    "booking_lead_time_days",
    "avg_price_range",  # computed from price_range
]

CATEGORICAL_COLS = [
    "cuisine",
    "location",
    "home_location",
    "dining_frequency",
    "favorite_cuisines",
    "preferred_",
    "dietary_res",
]

# ========================
# FastAPI app
# ========================

app = FastAPI(
    title="Restaurant Rating Prediction API",
    description="Predicts review ratings using restaurant, user, trend and review text features.",
    version="1.0.0",
)

# ========================
# Request / Response schemas
# ========================

class RatingRequest(BaseModel):
    # IDs included for convenience / logging (not used directly as features)
    restaurant_id: Optional[int] = Field(default=None, description="Restaurant ID")
    user_id: Optional[int] = Field(default=None, description="User ID")

    # ---- Text feature ----
    review_text: str = Field(description="Free-form review text")

    # ---- Restaurant features ----
    cuisine: Optional[str] = None
    location: Optional[str] = None
    price_range: Optional[str] = Field(
        default=None,
        description="Price range string, e.g. 'AED 50 - 100'"
    )
    rating_rest: Optional[float] = Field(
        default=None,
        description="Existing aggregate rating of the restaurant, e.g. 3.15"
    )
    review_count: Optional[int] = Field(
        default=None,
        description="Total number of reviews the restaurant has"
    )

    # ---- User features ----
    age: Optional[int] = None
    home_location: Optional[str] = None
    dining_frequency: Optional[str] = None
    favorite_cuisines: Optional[str] = None
    preferred_: Optional[str] = Field(
        default=None,
        description="Preference level / category from your dataset"
    )
    dietary_res: Optional[str] = Field(
        default=None,
        description="Dietary restrictions, e.g. Halal / None"
    )
    avg_rating_total_reviews_written: Optional[float] = Field(
        default=None,
        description="User's average rating over all reviews they've written"
    )

    # ---- Trend features (optional; you can pass them if you compute externally) ----
    popularity_score: Optional[float] = None
    avg_price: Optional[float] = None
    booking_lead_time_days: Optional[float] = None


class RatingResponse(BaseModel):
    restaurant_id: Optional[int]
    user_id: Optional[int]
    predicted_rating: float
    predicted_rating_rounded: float


class BatchRatingResponse(BaseModel):
    predictions: List[RatingResponse]


# ========================
# Helper functions
# ========================

def parse_price_range_to_avg(price_range: Optional[str]) -> Optional[float]:
    """
    Parse strings like 'AED 50 - 100' into a numeric average: 75.0
    Returns None if parsing fails.
    """
    if not isinstance(price_range, str):
        return None
    # Keep digits and spaces; replace others with space
    nums = "".join(ch if (ch.isdigit() or ch == " ") else " " for ch in price_range)
    parts = [p for p in nums.split() if p.isdigit()]
    if not parts:
        return None
    vals = list(map(float, parts))
    return float(np.mean(vals))


def build_feature_dataframe(payloads: List[RatingRequest]) -> pd.DataFrame:
    """
    Convert list of RatingRequest objects into a pandas DataFrame
    with the columns the model expects.
    """
    rows = []

    for item in payloads:
        data = item.dict()

        # Compute avg_price_range from price_range
        avg_price_range = parse_price_range_to_avg(data.get("price_range"))
        data["avg_price_range"] = avg_price_range

        # Only keep the columns the model expects:
        #   numeric + categorical + text
        row = {}

        # numeric
        for col in NUMERIC_COLS:
            row[col] = data.get(col, None)

        # categorical
        for col in CATEGORICAL_COLS:
            row[col] = data.get(col, None)

        # text
        row[TEXT_COL] = data.get(TEXT_COL, "")

        # Keep IDs separately for returning later (not as features)
        row["_restaurant_id"] = data.get("restaurant_id", None)
        row["_user_id"] = data.get("user_id", None)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Separate IDs (not used in model)
    id_cols = df[["_restaurant_id", "_user_id"]]
    features_df = df.drop(columns=["_restaurant_id", "_user_id"])

    return features_df, id_cols


# ========================
# Endpoints
# ========================

@app.get("/health", tags=["health"])
def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=BatchRatingResponse, tags=["prediction"])
def predict_ratings(requests: List[RatingRequest]):
    """
    Predict ratings for one or more reviews.
    """
    # Build feature DataFrame
    X, ids = build_feature_dataframe(requests)

    # Run model prediction
    preds = model.predict(X)

    responses = []
    for i, pred in enumerate(preds):
        restaurant_id = ids.iloc[i]["_restaurant_id"]
        user_id = ids.iloc[i]["_user_id"]
        rounded = float(np.round(pred, 2))  # round to 2 decimals

        responses.append(
            RatingResponse(
                restaurant_id=restaurant_id,
                user_id=user_id,
                predicted_rating=float(pred),
                predicted_rating_rounded=rounded,
            )
        )

    return BatchRatingResponse(predictions=responses)


# Optional: a simple root endpoint
@app.get("/", tags=["root"])
def root():
    return {
        "message": "Restaurant Rating Prediction API",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
    }
