from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Dict

class TransactionRequest(BaseModel):
    amt: float = Field(..., ge=0, description="Transaction amount in USD")
    gender: Literal["Male", "Female"] = Field(..., description="Gender of the cardholder")
    city_pop: int = Field(..., ge=0, description="Population of the city")
    month: int = Field(..., ge=1, le=12, description="Month of transaction (1-12)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    hours: int = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    age: int = Field(..., ge=0, le=120, description="Age of the cardholder")
    category: Literal[
        "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
        "health_fitness", "home", "kids_pets", "misc_net", "misc_pos", "personal_care",
        "shopping_net", "shopping_pos", "travel"
    ] = Field(..., description="Transaction category")
    merchant_freq: int = Field(..., ge=1, le=5000, description="Frequency encoding of merchant (1-5000)")
    job_freq: int = Field(..., ge=1, le=5000, description="Frequency encoding of job title (1-5000)")
    city_freq: int = Field(..., ge=1, le=5000, description="Frequency encoding of city (1-5000)")
    state_freq: int = Field(..., ge=1, le=5000, description="Frequency encoding of state (1-5000)")
    distance_km: float = Field(..., ge=0, description="Distance from home in kilometers")

    class Config:
        schema_extra = {
            "example": {
                "amt": 100.50,
                "gender": "Male",
                "city_pop": 50000,
                "month": 6,
                "day_of_week": 2,
                "hours": 14,
                "age": 35,
                "category": "grocery_net",
                "merchant_freq": 150,
                "job_freq": 200,
                "city_freq": 300,
                "state_freq": 20,
                "distance_km": 2.1
            }
        }

class TransactionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float = Field(..., ge=0, le=1)
    risk_level: Literal["Low", "Medium", "High"]
    feature_importances: Optional[List[Dict[str, float]]] = None