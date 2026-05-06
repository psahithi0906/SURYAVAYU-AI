from __future__ import annotations

import json
import math
import os
import random
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


MODEL_VERSION = "vasudha-lightgbm-lstm-v3"
MODEL_DIR = Path(__file__).resolve().parent / "models"
METRICS_CACHE_FILE = Path(__file__).resolve().parent / "metrics_cache.json"

# SCADA/EMS Integration Config
SCADA_API_URL = os.getenv("SCADA_API_URL", "https://api.kspdcl.com/scada/actuals")
SCADA_API_KEY = os.getenv("SCADA_API_KEY", "")
EMS_API_URL = os.getenv("EMS_API_URL", "https://api.kredl.com/ems/historical")
EMS_API_KEY = os.getenv("EMS_API_KEY", "")

FEATURE_NAMES = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "temperature_2m",
    "cloud_cover",
    "wind_speed_10m",
    "wind_speed_cubed",
    "wind_dir_sin",
    "wind_dir_cos",
    "shortwave_radiation",
    "capacity_mw",
    "is_solar",
    "is_wind",
    "is_hybrid",
]
_MODEL_CACHE: dict[str, Any] = {}
_LSTM_CACHE: dict[str, Any] = {}

METRICS_CACHE_FILE = Path(__file__).resolve().parent / "metrics_cache.json"


def _load_metrics_store() -> dict[str, Any]:
    if not METRICS_CACHE_FILE.exists():
        return {}
    try:
        with METRICS_CACHE_FILE.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _save_metrics_store() -> None:
    try:
        with METRICS_CACHE_FILE.open("w", encoding="utf-8") as handle:
            json.dump(_METRICS_STORE, handle, indent=2)
    except Exception:
        pass


def _update_metrics_store(plant_id: str, target_date: date, metrics: dict[str, Any]) -> None:
    entry = _METRICS_STORE.setdefault(plant_id, {})
    entry[target_date.isoformat()] = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    _save_metrics_store()


_METRICS_STORE: dict[str, Any] = _load_metrics_store()


@dataclass(frozen=True)
class Plant:
    plant_id: str
    name: str
    plant_type: str
    lat: float
    lng: float
    capacity_mw: float
    cluster_id: str


PLANTS: list[Plant] = [
    Plant("SOLAR_001", "Pavagada Solar Park", "solar", 14.0990, 77.2800, 2050, "Tumakuru-Solar"),
    Plant("SOLAR_002", "Raichur Solar Plains", "solar", 16.2120, 77.3439, 600, "Raichur-Solar"),
    Plant("SOLAR_003", "Koppal Solar Farm", "solar", 15.3480, 76.1540, 350, "Koppal-Solar"),
    Plant("SOLAR_004", "Kalaburagi Solar Zone", "solar", 17.3297, 76.8343, 520, "North-Solar"),
    Plant("SOLAR_005", "Yadgir Solar Field", "solar", 16.7700, 77.1376, 300, "North-Solar"),
    Plant("WIND_001", "Chitradurga Wind Farm", "wind", 14.2306, 76.3980, 450, "Central-Wind"),
    Plant("WIND_002", "Gadag Wind Corridor", "wind", 15.4310, 75.6350, 320, "Gadag-Wind"),
    Plant("WIND_003", "Vijayapura Wind Zone", "wind", 16.8302, 75.7100, 280, "North-Wind"),
    Plant("WIND_004", "Belagavi Wind Ridge", "wind", 15.8497, 74.4977, 500, "Western-Wind"),
    Plant("HYBRID_001", "Ballari Hybrid Park", "hybrid", 15.1394, 76.9214, 700, "Ballari-Hybrid"),
]

PLANT_BY_ID = {plant.plant_id: plant for plant in PLANTS}


def list_plants() -> list[dict[str, Any]]:
    return [
        {
            "plant_id": plant.plant_id,
            "name": plant.name,
            "plant_type": plant.plant_type,
            "lat": plant.lat,
            "lng": plant.lng,
            "capacity_mw": plant.capacity_mw,
            "cluster_id": plant.cluster_id,
        }
        for plant in PLANTS
    ]


def _fetch_open_meteo(plant: Plant, target_date: date, horizon_hours: int) -> dict[str, list[Any]]:
    end_date = target_date + timedelta(days=max(1, math.ceil(horizon_hours / 24)))
    params = {
        "latitude": plant.lat,
        "longitude": plant.lng,
        "start_date": target_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(
            [
                "temperature_2m",
                "cloud_cover",
                "wind_speed_10m",
                "wind_direction_10m",
                "shortwave_radiation",
                "relative_humidity_2m",
            ]
        ),
        "timezone": "Asia/Kolkata",
    }
    url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=8) as response:
        payload = response.read().decode("utf-8")

    import json

    data = json.loads(payload)
    hourly = data.get("hourly")
    if not hourly or "time" not in hourly:
        raise RuntimeError("Open-Meteo returned no hourly data")
    return hourly


def _synthetic_weather(plant: Plant, target_date: date, horizon_hours: int) -> dict[str, list[Any]]:
    rng = random.Random(f"{plant.plant_id}-{target_date.isoformat()}")
    times: list[str] = []
    temp: list[float] = []
    cloud: list[float] = []
    wind_speed: list[float] = []
    wind_dir: list[float] = []
    radiation: list[float] = []
    humidity: list[float] = []
    season = 0.5 + 0.5 * math.sin(2 * math.pi * (target_date.timetuple().tm_yday - 40) / 365)

    for i in range(horizon_hours):
        ts = datetime.combine(target_date, datetime.min.time()) + timedelta(hours=i)
        hour = ts.hour
        daylight = max(0, math.sin(math.pi * (hour - 6) / 12))
        cloud_val = min(96, max(4, 35 + 25 * math.sin(i / 5) + rng.uniform(-16, 16)))
        wind_val = max(1.2, 6.8 + 2.4 * math.sin(i / 4 + plant.lat) + rng.uniform(-1.3, 1.8))
        times.append(ts.isoformat(timespec="minutes"))
        temp.append(round(23 + 9 * daylight + 2 * season + rng.uniform(-1.5, 1.5), 2))
        cloud.append(round(cloud_val, 2))
        wind_speed.append(round(wind_val, 2))
        wind_dir.append(round((215 + 65 * math.sin(i / 7) + rng.uniform(-28, 28)) % 360, 1))
        radiation.append(round(980 * daylight * (1 - cloud_val / 140) * (0.76 + 0.18 * season), 2))
        humidity.append(round(min(95, max(25, 62 - 0.4 * temp[-1] + cloud_val / 5)), 2))

    return {
        "time": times,
        "temperature_2m": temp,
        "cloud_cover": cloud,
        "wind_speed_10m": wind_speed,
        "wind_direction_10m": wind_dir,
        "shortwave_radiation": radiation,
        "relative_humidity_2m": humidity,
    }


def _weather(plant: Plant, target_date: date, horizon_hours: int) -> tuple[dict[str, list[Any]], str]:
    hourly = _fetch_open_meteo(plant, target_date, horizon_hours)
    if len(hourly["time"]) < horizon_hours:
        raise RuntimeError("Insufficient weather data from Open-Meteo")
    return hourly, "open-meteo"


def _solar_mw(radiation: float, cloud: float, temp: float, capacity: float) -> float:
    temp_loss = max(0.82, 1 - max(0, temp - 25) * 0.004)
    cloud_factor = max(0, 1 - cloud / 120)
    return max(0, min(capacity, capacity * (radiation / 1000) * cloud_factor * temp_loss))


def _wind_mw(speed: float, capacity: float) -> float:
    if speed < 3 or speed > 25:
        return 0
    if speed < 12:
        return capacity * ((speed - 3) / 9) ** 3
    return capacity


def _hybrid_mw(row: dict[str, float], capacity: float) -> float:
    return _solar_mw(row["radiation"], row["cloud"], row["temp"], capacity * 0.58) + _wind_mw(
        row["wind_speed"], capacity * 0.42
    )


def _feature_vector(plant: Plant, timestamp: str, row: dict[str, float]) -> list[float]:
    dt = datetime.fromisoformat(timestamp)
    hour_angle = 2 * math.pi * dt.hour / 24
    month_angle = 2 * math.pi * dt.month / 12
    wind_dir_rad = math.radians(row["wind_dir"])
    return [
        math.sin(hour_angle),
        math.cos(hour_angle),
        math.sin(month_angle),
        math.cos(month_angle),
        row["temp"],
        row["cloud"],
        row["wind_speed"],
        row["wind_speed"] ** 3,
        math.sin(wind_dir_rad),
        math.cos(wind_dir_rad),
        row["radiation"],
        plant.capacity_mw,
        1.0 if plant.plant_type == "solar" else 0.0,
        1.0 if plant.plant_type == "wind" else 0.0,
        1.0 if plant.plant_type == "hybrid" else 0.0,
    ]


def _load_model(plant_type: str) -> Any | None:
    if plant_type in _MODEL_CACHE:
        return _MODEL_CACHE[plant_type]
    try:
        import joblib

        artifact = MODEL_DIR / f"{plant_type}_forecast_model.joblib"
        if not artifact.exists():
            _MODEL_CACHE[plant_type] = None
            return None
        bundle = joblib.load(artifact)
        _MODEL_CACHE[plant_type] = bundle
        return bundle
    except Exception:
        _MODEL_CACHE[plant_type] = None
        return None


def _load_lstm_model(plant_type: str) -> Any | None:
    if plant_type in _LSTM_CACHE:
        return _LSTM_CACHE[plant_type]
    try:
        try:
            from modeling import RenewableLSTM, torch
        except ModuleNotFoundError:
            from backend.modeling import RenewableLSTM, torch

        if torch is None or RenewableLSTM is None:
            _LSTM_CACHE[plant_type] = None
            return None
        artifact = MODEL_DIR / f"{plant_type}_lstm_intraday.pt"
        if not artifact.exists():
            _LSTM_CACHE[plant_type] = None
            return None
        bundle = torch.load(artifact, map_location="cpu", weights_only=False)
        model = RenewableLSTM(input_size=bundle["input_size"], hidden_size=bundle["hidden_size"])
        model.load_state_dict(bundle["state_dict"])
        model.eval()
        bundle["model"] = model
        bundle["torch"] = torch
        _LSTM_CACHE[plant_type] = bundle
        return bundle
    except Exception:
        _LSTM_CACHE[plant_type] = None
        return None


def _formula_prediction(plant: Plant, row: dict[str, float]) -> float:
    if plant.plant_type == "solar":
        return _solar_mw(row["radiation"], row["cloud"], row["temp"], plant.capacity_mw)
    if plant.plant_type == "wind":
        return _wind_mw(row["wind_speed"], plant.capacity_mw)
    return _hybrid_mw(row, plant.capacity_mw)


def _model_prediction(
    plant: Plant,
    timestamp: str,
    row: dict[str, float],
    forecast_type: str = "day_ahead",
    sequence_features: list[list[float]] | None = None,
) -> tuple[float, str]:
    if forecast_type == "intraday":
        lstm_bundle = _load_lstm_model(plant.plant_type)
        if lstm_bundle and sequence_features:
            try:
                torch = lstm_bundle["torch"]
                seq = sequence_features[-6:]
                while len(seq) < 6:
                    seq.insert(0, seq[0])
                if "feature_mean" in lstm_bundle and "feature_std" in lstm_bundle:
                    seq = [
                        [
                            (value - float(lstm_bundle["feature_mean"][feature_index]))
                            / float(lstm_bundle["feature_std"][feature_index])
                            for feature_index, value in enumerate(feature_row)
                        ]
                        for feature_row in seq
                    ]
                tensor = torch.tensor([seq], dtype=torch.float32)
                with torch.no_grad():
                    normalized = float(lstm_bundle["model"](tensor).clamp(0, 1).item())
                p50 = max(0, min(plant.capacity_mw, normalized * plant.capacity_mw))
                if plant.plant_type == "solar" and row["radiation"] < 10:
                    p50 = 0
                if plant.plant_type == "hybrid" and row["radiation"] < 10:
                    p50 = min(p50, _wind_mw(row["wind_speed"], plant.capacity_mw * 0.42))
                return p50, lstm_bundle.get("model_name", "pytorch-lstm-intraday")
            except Exception:
                pass

    bundle = _load_model(plant.plant_type)
    if not bundle:
        return _formula_prediction(plant, row), "physics-fallback"
    try:
        features = [_feature_vector(plant, timestamp, row)]
        normalized = float(bundle["model"].predict(features)[0])
        p50 = max(0, min(plant.capacity_mw, normalized * plant.capacity_mw))
        return p50, bundle.get("model_name", "lightgbm-weather-model")
    except Exception:
        return _formula_prediction(plant, row), "physics-fallback"


def _drivers(plant: Plant, rows: list[dict[str, float]]) -> list[dict[str, Any]]:
    avg_cloud = sum(row["cloud"] for row in rows) / len(rows)
    avg_wind = sum(row["wind_speed"] for row in rows) / len(rows)
    avg_rad = sum(row["radiation"] for row in rows) / len(rows)
    avg_temp = sum(row["temp"] for row in rows) / len(rows)
    drivers = [
        ("Shortwave radiation", (avg_rad - 420) / 420, "increase"),
        ("Cloud cover", -(avg_cloud - 35) / 70, "decrease" if avg_cloud > 35 else "increase"),
        ("Wind speed cubed", ((avg_wind**3) - (6.5**3)) / (12**3), "increase"),
        ("Temperature derate", -max(0, avg_temp - 25) / 22, "decrease"),
        ("Diurnal seasonality", 0.18, "increase"),
    ]
    if plant.plant_type == "solar":
        selected = [drivers[i] for i in [0, 1, 3, 4, 2]]
    elif plant.plant_type == "wind":
        selected = [drivers[i] for i in [2, 1, 4, 3, 0]]
    else:
        selected = drivers
    return [
        {"feature": feature, "value": round(value, 3), "direction": direction}
        for feature, value, direction in selected[:5]
    ]


def forecast(plant_id: str, target_date: date, horizon_hours: int = 24, forecast_type: str = "day_ahead") -> dict[str, Any]:
    if plant_id not in PLANT_BY_ID:
        raise KeyError(f"Unknown plant_id: {plant_id}")
    horizon = min(max(horizon_hours, 1), 72)
    plant = PLANT_BY_ID[plant_id]
    points, model_source, source, rows = _forecast_points(plant, target_date, horizon, forecast_type)
    metrics, metrics_source = _compute_metrics_from_history(plant, target_date, horizon, forecast_type)
    _update_metrics_store(plant_id, target_date, metrics)

    return {
        "plant_id": plant.plant_id,
        "plant_name": plant.name,
        "plant_type": plant.plant_type,
        "capacity_mw": plant.capacity_mw,
        "cluster_id": plant.cluster_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "forecast_type": forecast_type,
        "model_version": MODEL_VERSION,
        "model_source": model_source if points else "not-run",
        "weather_source": source,
        "points": points,
        "top_drivers": _drivers(plant, rows),
        "metrics": metrics,
        "metrics_source": metrics_source,
    }


def _nrmse(points: list[dict[str, Any]], actual_points: list[dict[str, Any]], capacity: float) -> float:
    if not points:
        return 0.0
    mse = sum((float(actual["actual_mw"]) - float(point["p50"])) ** 2 for point, actual in zip(points, actual_points)) / len(points)
    return math.sqrt(mse) / max(capacity, 1.0)


def _coverage_p10_p90(points: list[dict[str, Any]], actual_points: list[dict[str, Any]]) -> float:
    if not points:
        return 0.0
    inside = sum(1 for point, actual in zip(points, actual_points) if point["p10"] <= actual["actual_mw"] <= point["p90"])
    return inside / len(points)


def _pinball_loss_p90(points: list[dict[str, Any]], actual_points: list[dict[str, Any]]) -> float:
    if not points:
        return 0.0
    q = 0.9
    loss = 0.0
    for point, actual in zip(points, actual_points):
        diff = float(actual["actual_mw"]) - float(point["p90"])
        loss += max(q * diff, (q - 1) * diff)
    return loss / len(points)


def _baseline_improvement(points: list[dict[str, Any]], actual_points: list[dict[str, Any]]) -> float:
    if len(points) < 2:
        return 0.0
    baseline_errors = []
    model_errors = []
    for idx, (point, actual) in enumerate(zip(points, actual_points)):
        actual_value = float(actual["actual_mw"])
        model_errors.append((actual_value - float(point["p50"])) ** 2)
        baseline_value = float(actual_points[idx - 1]["actual_mw"]) if idx > 0 else actual_value
        baseline_errors.append((actual_value - baseline_value) ** 2)
    rmse_model = math.sqrt(sum(model_errors) / len(model_errors))
    rmse_baseline = math.sqrt(sum(baseline_errors) / len(baseline_errors))
    if rmse_baseline <= 0:
        return 0.0
    return max(-1.0, 1 - rmse_model / rmse_baseline)


def _compute_metrics_from_history(plant: Plant, target_date: date, horizon_hours: int, forecast_type: str) -> tuple[dict[str, Any], str]:
    history_date = target_date - timedelta(days=1)
    try:
        points, _, _, _ = _forecast_points(plant, history_date, horizon_hours, forecast_type)
        actual_points = _fetch_ems_historical_actuals(plant, history_date, horizon_hours)
        if len(points) != len(actual_points):
            raise ValueError("Historical actuals length mismatch")
        metrics = {
            "nrmse": round(_nrmse(points, actual_points, plant.capacity_mw), 3),
            "coverage_p10_p90": round(_coverage_p10_p90(points, actual_points), 3),
            "baseline_improvement": round(_baseline_improvement(points, actual_points), 3),
            "pinball_loss_p90": round(_pinball_loss_p90(points, actual_points), 3),
        }
        return metrics, "ems-historical"
    except Exception as e:
        # No fallback - raise error for real-time requirement
        raise RuntimeError(f"Failed to fetch real historical data from EMS: {e}")


def _fetch_scada_actuals(plant: Plant, target_date: date, hours: int) -> list[dict[str, Any]]:
    """Fetch real-time actuals from SCADA system."""
    import requests
    headers = {"Authorization": f"Bearer {SCADA_API_KEY}"} if SCADA_API_KEY else {}
    params = {
        "plant_id": plant.plant_id,
        "start_date": target_date.isoformat(),
        "hours": hours,
    }
    response = requests.get(SCADA_API_URL, headers=headers, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    # Expected format: [{"timestamp": "2026-05-05T10:00:00", "actual_mw": 150.5}, ...]
    return data


def _fetch_ems_historical_actuals(plant: Plant, target_date: date, hours: int) -> list[dict[str, Any]]:
    """Fetch historical actuals from EMS for KPI computation."""
    import requests
    headers = {"Authorization": f"Bearer {EMS_API_KEY}"} if EMS_API_KEY else {}
    params = {
        "plant_id": plant.plant_id,
        "date": target_date.isoformat(),
        "hours": hours,
    }
    response = requests.get(EMS_API_URL, headers=headers, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    return data


def _forecast_points(plant: Plant, target_date: date, horizon_hours: int, forecast_type: str = "day_ahead") -> tuple[list[dict[str, Any]], str, str, list[dict[str, float]]]:
    hourly, source = _weather(plant, target_date, horizon_hours)
    rows: list[dict[str, float]] = []
    points: list[dict[str, Any]] = []
    model_source = "not-run"
    for idx, ts in enumerate(hourly["time"][:horizon_hours]):
        row = {
            "temp": float(hourly.get("temperature_2m", [28] * horizon_hours)[idx] or 28),
            "cloud": float(hourly.get("cloud_cover", [35] * horizon_hours)[idx] or 35),
            "wind_speed": float(hourly.get("wind_speed_10m", [6] * horizon_hours)[idx] or 6),
            "wind_dir": float(hourly.get("wind_direction_10m", [220] * horizon_hours)[idx] or 220),
            "radiation": float(hourly.get("shortwave_radiation", [0] * horizon_hours)[idx] or 0),
        }
        rows.append(row)
        sequence_so_far = [_feature_vector(plant, prior_ts, prior_row) for prior_ts, prior_row in zip(hourly["time"][: idx + 1], rows)]
        p50, model_source = _model_prediction(plant, ts, row, forecast_type, sequence_so_far)
        weather_risk = 0.08 + (row["cloud"] / 100) * 0.11 + min(0.12, abs(row["wind_speed"] - 8) / 90)
        spread = max(plant.capacity_mw * 0.012, p50 * weather_risk)
        p10 = max(0, p50 - 1.28 * spread)
        p90 = min(plant.capacity_mw, p50 + 1.28 * spread)
        points.append(
            {
                "timestamp": ts,
                "p10": round(p10, 2),
                "p50": round(p50, 2),
                "p90": round(max(p90, p50 + 0.01), 2),
                "weather": {
                    "temperature_2m": row["temp"],
                    "cloud_cover": row["cloud"],
                    "wind_speed_10m": row["wind_speed"],
                    "wind_direction_10m": row["wind_dir"],
                    "shortwave_radiation": row["radiation"],
                },
            }
        )
    return points, model_source, source, rows


def actuals(plant_id: str, target_date: date, hours: int = 24) -> list[dict[str, Any]]:
    """Fetch real-time actuals from SCADA - no synthetic fallbacks."""
    if plant_id not in PLANT_BY_ID:
        raise KeyError(f"Unknown plant_id: {plant_id}")
    plant = PLANT_BY_ID[plant_id]
    # Fetch real data - fail if not available
    return _fetch_scada_actuals(plant, target_date, hours)
