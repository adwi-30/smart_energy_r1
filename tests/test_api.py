from fastapi.testclient import TestClient
from api import app

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "message": "Smart Energy RL API is running"}

def test_predict_action():
    with TestClient(app) as client:
        payload = {
            "hour": 14,
            "occupancies": [2, 1, 0, 0],
            "temperatures": [1, 1, 1, 1]
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "hvac_on" in data
        assert "lighting_on" in data
        assert len(data["hvac_on"]) == 4
        assert len(data["lighting_on"]) == 4
