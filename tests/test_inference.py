import os
import pytest

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def upload_file(client, token, filename):
    filepath = os.path.join(FIXTURES_DIR, filename)
    with open(filepath, "rb") as f:
        resp = client.post(
            "/api/files/upload",
            files={"file": (filename, f, "text/csv")},
            headers=auth_headers(token),
        )
    assert resp.status_code == 200, resp.text
    return resp.json()["file_id"]


def train_model(client, token, file_id, target, task_type, algorithm="random_forest"):
    resp = client.post(
        "/api/training/train",
        json={
            "file_id": file_id,
            "target_column": target,
            "task_type": task_type,
            "algorithm": algorithm,
        },
        headers=auth_headers(token),
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["job_id"]


# ---------------------------------------------------------------------------
# Happy path — classification
# ---------------------------------------------------------------------------

def test_predict_random_forest_classification(client, auth_token):
    train_file_id = upload_file(client, auth_token, "classification.csv")
    job_id = train_model(client, auth_token, train_file_id, "label", "classification", "random_forest")
    infer_file_id = upload_file(client, auth_token, "inference_input.csv")

    resp = client.post(
        "/api/inference/predict",
        json={"job_id": job_id, "file_id": infer_file_id},
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["task_type"] == "classification"
    assert data["algorithm"] == "random_forest"
    assert data["num_rows"] == 10
    assert len(data["predictions"]) == 10
    assert all(isinstance(p, int) for p in data["predictions"])
    assert len(data["data_with_predictions"]) == 10
    assert "prediction" in data["data_with_predictions"][0]


def test_predict_lightgbm_classification(client, auth_token):
    train_file_id = upload_file(client, auth_token, "classification.csv")
    job_id = train_model(client, auth_token, train_file_id, "label", "classification", "lightgbm")
    infer_file_id = upload_file(client, auth_token, "inference_input.csv")

    resp = client.post(
        "/api/inference/predict",
        json={"job_id": job_id, "file_id": infer_file_id},
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["num_rows"] == 10


def test_predict_xgboost_classification(client, auth_token):
    train_file_id = upload_file(client, auth_token, "classification.csv")
    job_id = train_model(client, auth_token, train_file_id, "label", "classification", "xgboost")
    infer_file_id = upload_file(client, auth_token, "inference_input.csv")

    resp = client.post(
        "/api/inference/predict",
        json={"job_id": job_id, "file_id": infer_file_id},
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["num_rows"] == 10


# ---------------------------------------------------------------------------
# Happy path — regression
# ---------------------------------------------------------------------------

def test_predict_random_forest_regression(client, auth_token):
    train_file_id = upload_file(client, auth_token, "regression.csv")
    job_id = train_model(client, auth_token, train_file_id, "value", "regression", "random_forest")
    infer_file_id = upload_file(client, auth_token, "inference_input.csv")

    resp = client.post(
        "/api/inference/predict",
        json={"job_id": job_id, "file_id": infer_file_id},
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["task_type"] == "regression"
    assert data["num_rows"] == 10
    assert all(isinstance(p, float) for p in data["predictions"])


def test_predict_lightgbm_regression(client, auth_token):
    train_file_id = upload_file(client, auth_token, "regression.csv")
    job_id = train_model(client, auth_token, train_file_id, "value", "regression", "lightgbm")
    infer_file_id = upload_file(client, auth_token, "inference_input.csv")

    resp = client.post(
        "/api/inference/predict",
        json={"job_id": job_id, "file_id": infer_file_id},
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["num_rows"] == 10


def test_predict_xgboost_regression(client, auth_token):
    train_file_id = upload_file(client, auth_token, "regression.csv")
    job_id = train_model(client, auth_token, train_file_id, "value", "regression", "xgboost")
    infer_file_id = upload_file(client, auth_token, "inference_input.csv")

    resp = client.post(
        "/api/inference/predict",
        json={"job_id": job_id, "file_id": infer_file_id},
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["num_rows"] == 10


# ---------------------------------------------------------------------------
# Column validation
# ---------------------------------------------------------------------------

def test_predict_column_mismatch(client, auth_token):
    train_file_id = upload_file(client, auth_token, "classification.csv")
    job_id = train_model(client, auth_token, train_file_id, "label", "classification")
    bad_file_id = upload_file(client, auth_token, "inference_bad_columns.csv")

    resp = client.post(
        "/api/inference/predict",
        json={"job_id": job_id, "file_id": bad_file_id},
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 400
    assert "missing required columns" in resp.json()["detail"]


def test_predict_extra_columns_ok(client, auth_token):
    """CSV with extra columns beyond f1/f2/f3 should still work."""
    import csv, io
    # Build a CSV with f1, f2, f3, f4 (extra column)
    content = "f1,f2,f3,f4\n"
    for _ in range(10):
        content += "1.0,2.0,3.0,99.0\n"

    train_file_id = upload_file(client, auth_token, "classification.csv")
    job_id = train_model(client, auth_token, train_file_id, "label", "classification")

    resp = client.post(
        "/api/files/upload",
        files={"file": ("extra.csv", content.encode(), "text/csv")},
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 200
    extra_file_id = resp.json()["file_id"]

    resp = client.post(
        "/api/inference/predict",
        json={"job_id": job_id, "file_id": extra_file_id},
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["num_rows"] == 10


def test_predict_non_numeric_input(client, auth_token):
    train_file_id = upload_file(client, auth_token, "classification.csv")
    job_id = train_model(client, auth_token, train_file_id, "label", "classification")

    # Build CSV with f1, f2, f3 but f1 is non-numeric
    content = "f1,f2,f3\nfoo,1.0,2.0\nbar,2.0,3.0\n"
    resp = client.post(
        "/api/files/upload",
        files={"file": ("nonnum.csv", content.encode(), "text/csv")},
        headers=auth_headers(auth_token),
    )
    bad_file_id = resp.json()["file_id"]

    resp = client.post(
        "/api/inference/predict",
        json={"job_id": job_id, "file_id": bad_file_id},
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 400
    assert "Non-numeric" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_predict_job_not_found(client, auth_token):
    infer_file_id = upload_file(client, auth_token, "inference_input.csv")
    resp = client.post(
        "/api/inference/predict",
        json={"job_id": "nonexistent-job-id", "file_id": infer_file_id},
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 404


def test_predict_file_not_found(client, auth_token):
    train_file_id = upload_file(client, auth_token, "classification.csv")
    job_id = train_model(client, auth_token, train_file_id, "label", "classification")

    resp = client.post(
        "/api/inference/predict",
        json={"job_id": job_id, "file_id": "nonexistent-file-id"},
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 404


def test_predict_empty_csv(client, auth_token):
    train_file_id = upload_file(client, auth_token, "classification.csv")
    job_id = train_model(client, auth_token, train_file_id, "label", "classification")

    content = "f1,f2,f3\n"
    resp = client.post(
        "/api/files/upload",
        files={"file": ("empty.csv", content.encode(), "text/csv")},
        headers=auth_headers(auth_token),
    )
    empty_file_id = resp.json()["file_id"]

    resp = client.post(
        "/api/inference/predict",
        json={"job_id": job_id, "file_id": empty_file_id},
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------

def test_predict_unauthenticated(client):
    resp = client.post(
        "/api/inference/predict",
        json={"job_id": "any-id", "file_id": "any-id"},
    )
    assert resp.status_code == 403


def test_predict_other_users_job(client, auth_token):
    train_file_id = upload_file(client, auth_token, "classification.csv")
    job_id = train_model(client, auth_token, train_file_id, "label", "classification")

    client.post(
        "/api/auth/register",
        json={"username": "infer_other", "email": "infer_other@example.com", "password": "password123"},
    )
    other_token = client.post(
        "/api/auth/login",
        json={"username": "infer_other", "password": "password123"},
    ).json()["access_token"]

    # other user uploads their own file
    other_file_id = upload_file(client, other_token, "inference_input.csv")

    resp = client.post(
        "/api/inference/predict",
        json={"job_id": job_id, "file_id": other_file_id},
        headers=auth_headers(other_token),
    )
    assert resp.status_code == 403


def test_predict_other_users_file(client, auth_token):
    train_file_id = upload_file(client, auth_token, "classification.csv")
    job_id = train_model(client, auth_token, train_file_id, "label", "classification")

    client.post(
        "/api/auth/register",
        json={"username": "file_owner", "email": "file_owner@example.com", "password": "password123"},
    )
    other_token = client.post(
        "/api/auth/login",
        json={"username": "file_owner", "password": "password123"},
    ).json()["access_token"]
    other_file_id = upload_file(client, other_token, "inference_input.csv")

    # user 1 tries to predict using user 2's file
    resp = client.post(
        "/api/inference/predict",
        json={"job_id": job_id, "file_id": other_file_id},
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 403
