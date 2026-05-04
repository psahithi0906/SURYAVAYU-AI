from __future__ import annotations

from datetime import date as dt_date, datetime, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from forecast_engine import MODEL_VERSION, actuals, forecast, list_plants


class ForecastRequest(BaseModel):
    plant_id: str = Field(examples=["SOLAR_001"])
    date: dt_date = Field(default_factory=dt_date.today)
    horizon_hours: int = Field(default=24, ge=1, le=72)
    forecast_type: Literal["day_ahead", "intraday"] = "day_ahead"


app = FastAPI(
    title="VASUDHA AI",
    version="1.0.0",
    description="Open-Meteo powered renewable energy forecasting API for Karnataka solar and wind assets.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_versions": {"forecast": MODEL_VERSION},
        "last_trained_at": "2026-05-04T00:00:00+00:00",
        "data_freshness": datetime.now(timezone.utc).isoformat(),
        "weather_provider": "Open-Meteo forecast API with deterministic fallback",
    }


@app.get("/plants")
def plants() -> list[dict]:
    return list_plants()


@app.post("/forecast")
def post_forecast(payload: ForecastRequest) -> dict:
    try:
        return forecast(payload.plant_id, payload.date, payload.horizon_hours, payload.forecast_type)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/forecast/{plant_id}")
def get_forecast(
    plant_id: str,
    target_date: dt_date = Query(default_factory=dt_date.today, alias="date"),
    horizon_hours: int = Query(default=24, ge=1, le=72),
) -> dict:
    try:
        return forecast(plant_id, target_date, horizon_hours)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/actuals/{plant_id}")
def get_actuals(
    plant_id: str,
    target_date: dt_date = Query(default_factory=dt_date.today, alias="date"),
    hours: int = Query(default=24, ge=1, le=72),
) -> list[dict]:
    try:
        return actuals(plant_id, target_date, hours)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
