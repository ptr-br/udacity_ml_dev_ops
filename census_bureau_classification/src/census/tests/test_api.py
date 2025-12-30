# tests/test_api.py
from fastapi.testclient import TestClient
from census.main import app

def test_root_get():
    # Use context manager so FastAPI lifespan (artifact loading) runs
    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("message") == "Welcome to the FastAPI model inference service."

def test_post_predict_le_50k():
    payload = {
        "age": 50,
        "workclass": "State-gov",
        "fnlgt": 234721,
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Separated",
        "occupation": "Exec-managerial",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States",
    }
    with TestClient(app) as client:
        resp = client.post("/infer", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "prediction" in data
        assert data["prediction"] == "<=50K"

def test_post_predict_gt_50k():
    payload = {
        "age": 28,
        "workclass": "Private",
        "fnlgt": 123456,
        "education": "Doctorate",
        "education-num": 12,
        "marital-status": "Separated",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 10,
        "capital-loss": 5,
        "hours-per-week": 35,
        "native-country": "United-States",
    }
    with TestClient(app) as client:
        resp = client.post("/infer", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "prediction" in data
        assert data["prediction"] == ">50K"