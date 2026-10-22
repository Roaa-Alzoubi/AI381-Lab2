from fastapi.testclient import TestClient

from src.app.main import app, model

client = TestClient(app)

VALID_CUSTOMER = {
    "tenure_months": 5,
    "monthly_charges": 89.9,
    "contract": "month-to-month",
    "internet_service": "fiber",
    "support_calls": 3,
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_prediction():
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    probability = response.json()["churn_probability"]
    assert isinstance(probability, (int, float))
    assert 0 <= probability <= 1


def test_model_is_loaded():
    assert model is not None
    assert hasattr(model, "predict_proba")


def test_missing_input_is_rejected():
    customer = VALID_CUSTOMER.copy()
    del customer["tenure_months"]

    response = client.post("/predict", json=customer)
    assert response.status_code == 422


def test_negative_input_is_rejected():
    customer = VALID_CUSTOMER.copy()
    customer["tenure_months"] = -1

    response = client.post("/predict", json=customer)
    assert response.status_code == 422


def test_prediction_is_deterministic():
    first = client.post("/predict", json=VALID_CUSTOMER)
    second = client.post("/predict", json=VALID_CUSTOMER)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
