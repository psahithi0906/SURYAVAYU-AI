import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import app


client = TestClient(app)


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_plants_available():
    response = client.get("/plants")
    assert response.status_code == 200
    assert len(response.json()) == 10


def test_forecast_bands_ordered():
    response = client.post(
        "/forecast",
        json={"plant_id": "SOLAR_001", "date": "2026-05-04", "horizon_hours": 24},
    )
    assert response.status_code == 200
    points = response.json()["points"]
    assert len(points) == 24
    assert all(point["p10"] <= point["p50"] <= point["p90"] for point in points)


def test_invalid_plant_returns_404():
    response = client.post(
        "/forecast",
        json={"plant_id": "BAD_PLANT", "date": "2026-05-04", "horizon_hours": 24},
    )
    assert response.status_code == 404
