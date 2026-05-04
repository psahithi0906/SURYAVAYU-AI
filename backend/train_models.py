from __future__ import annotations

import json
import math
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from modeling import RenewableLSTM, torch

from forecast_engine import (
    FEATURE_NAMES,
    MODEL_DIR,
    PLANTS,
    Plant,
    _feature_vector,
    _formula_prediction,
)


def synthetic_row(plant: Plant, ts: datetime, rng: random.Random) -> dict[str, float]:
    hour = ts.hour + ts.minute / 60
    day_of_year = ts.timetuple().tm_yday
    season = 0.5 + 0.5 * math.sin(2 * math.pi * (day_of_year - 42) / 365)
    daylight = max(0, math.sin(math.pi * (hour - 6) / 12))
    cloud = min(98, max(2, rng.gauss(38, 24)))
    wind_speed = max(0.5, rng.gauss(7.3 + 1.8 * math.sin(day_of_year / 30 + plant.lat), 2.5))
    radiation = max(0, 1020 * daylight * (1 - cloud / 145) * (0.74 + 0.22 * season) + rng.gauss(0, 25))
    temp = 22 + 11 * daylight + 4 * season + rng.gauss(0, 2.1)
    wind_dir = (220 + 75 * math.sin(day_of_year / 25) + rng.gauss(0, 38)) % 360
    return {
        "temp": temp,
        "cloud": cloud,
        "wind_speed": wind_speed,
        "wind_dir": wind_dir,
        "radiation": radiation,
    }


def build_dataset(plant_type: str, samples_per_plant: int = 450) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(f"vasudha-{plant_type}-training")
    rows: list[list[float]] = []
    targets: list[float] = []
    plants = [plant for plant in PLANTS if plant.plant_type == plant_type]
    start = date(2024, 1, 1)

    for plant in plants:
        for _ in range(samples_per_plant):
            offset_days = rng.randrange(0, 730)
            hour = rng.randrange(0, 24)
            ts = datetime.combine(start + timedelta(days=offset_days), datetime.min.time()) + timedelta(hours=hour)
            weather = synthetic_row(plant, ts, rng)
            generation = _formula_prediction(plant, weather)
            noisy_generation = max(0, generation * (1 + rng.gauss(0, 0.035)))
            rows.append(_feature_vector(plant, ts.isoformat(timespec="minutes"), weather))
            targets.append(min(1, noisy_generation / plant.capacity_mw))

    return np.array(rows, dtype=float), np.array(targets, dtype=float)


def train_one(plant_type: str) -> dict[str, float | str | int]:
    x, y = build_dataset(plant_type)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.18, random_state=42, shuffle=True)
    model = LGBMRegressor(
        objective="regression",
        n_estimators=90,
        learning_rate=0.06,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=18,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    pred = np.clip(model.predict(x_test), 0, 1)
    rmse = math.sqrt(mean_squared_error(y_test, pred))
    metrics = {
        "plant_type": plant_type,
        "samples": int(len(y)),
        "nrmse": round(float(rmse), 4),
        "mae": round(float(mean_absolute_error(y_test, pred)), 4),
        "r2": round(float(r2_score(y_test, pred)), 4),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "metrics": metrics,
            "model_name": f"lightgbm-{plant_type}-day-ahead-v1",
            "model_family": "LightGBM",
        },
        MODEL_DIR / f"{plant_type}_forecast_model.joblib",
    )
    return metrics


def build_lstm_sequences(plant_type: str, days_per_plant: int = 12) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(f"vasudha-{plant_type}-lstm")
    x: list[list[list[float]]] = []
    y: list[float] = []
    plants = [plant for plant in PLANTS if plant.plant_type == plant_type]
    start = date(2024, 1, 1)

    for plant in plants:
        for day_index in range(days_per_plant):
            day = start + timedelta(days=day_index * 3)
            sequence: list[list[float]] = []
            target: list[float] = []
            for hour in range(24):
                ts = datetime.combine(day, datetime.min.time()) + timedelta(hours=hour)
                weather = synthetic_row(plant, ts, rng)
                sequence.append(_feature_vector(plant, ts.isoformat(timespec="minutes"), weather))
                target.append(min(1, _formula_prediction(plant, weather) / plant.capacity_mw))
            for end in range(6, 24):
                x.append(sequence[max(0, end - 6) : end])
                y.append(target[end])

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


def train_lstm_one(plant_type: str) -> dict[str, float | str | int]:
    if torch is None or RenewableLSTM is None:
        return {"plant_type": plant_type, "model_family": "LSTM", "status": "torch-unavailable"}
    torch.set_num_threads(1)

    x, y = build_lstm_sequences(plant_type)
    split = int(len(y) * 0.82)
    mean = x[:split].reshape(-1, x.shape[-1]).mean(axis=0)
    std = x[:split].reshape(-1, x.shape[-1]).std(axis=0) + 1e-6
    x = (x - mean) / std

    x_train = torch.tensor(x[:split])
    y_train = torch.tensor(y[:split])
    x_test = torch.tensor(x[split:])
    y_test = torch.tensor(y[split:])

    model = RenewableLSTM(input_size=x.shape[-1], hidden_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    loss_fn = torch.nn.MSELoss()
    model.train()
    for _ in range(35):
        optimizer.zero_grad()
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(x_test).clamp(0, 1).numpy()
    truth = y_test.numpy()
    rmse = math.sqrt(mean_squared_error(truth, pred))
    metrics = {
        "plant_type": plant_type,
        "model_family": "LSTM",
        "samples": int(len(y)),
        "nrmse": round(float(rmse), 4),
        "mae": round(float(mean_absolute_error(truth, pred)), 4),
        "r2": round(float(r2_score(truth, pred)), 4),
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "metrics": metrics,
            "model_name": f"pytorch-lstm-{plant_type}-intraday-v1",
            "input_size": int(x.shape[-1]),
            "hidden_size": 32,
            "feature_mean": mean.astype(np.float32),
            "feature_std": std.astype(np.float32),
        },
        MODEL_DIR / f"{plant_type}_lstm_intraday.pt",
    )
    return metrics


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    lightgbm_metrics = [train_one(plant_type) for plant_type in ["solar", "wind", "hybrid"]]
    lstm_metrics = [train_lstm_one(plant_type) for plant_type in ["solar", "wind", "hybrid"]]
    all_metrics = {"lightgbm": lightgbm_metrics, "lstm": lstm_metrics}
    metrics_path = MODEL_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(json.dumps(all_metrics, indent=2))
    print(f"Saved models to {MODEL_DIR}")


if __name__ == "__main__":
    main()
