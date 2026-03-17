import io
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


# ---------------------------------------------------------------------------
# Happy path — classification
# ---------------------------------------------------------------------------

def test_train_random_forest_classification(client, auth_token):
    file_id = upload_file(client, auth_token, "classification.csv")
    resp = client.post(
        "/api/training/train",
        json={
            "file_id": file_id,
            "target_column": "label",
            "task_type": "classification",
            "algorithm": "random_forest",
        },
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "completed"
    assert "accuracy" in data["metrics"]
    assert "precision" in data["metrics"]
    assert "recall" in data["metrics"]
    assert "f1_score" in data["metrics"]
    assert "confusion_matrix" in data["metrics"]
    assert data["training_duration_seconds"] > 0


def test_train_lightgbm_classification(client, auth_token):
    file_id = upload_file(client, auth_token, "classification.csv")
    resp = client.post(
        "/api/training/train",
        json={
            "file_id": file_id,
            "target_column": "label",
            "task_type": "classification",
            "algorithm": "lightgbm",
        },
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "completed"


def test_train_xgboost_classification(client, auth_token):
    file_id = upload_file(client, auth_token, "classification.csv")
    resp = client.post(
        "/api/training/train",
        json={
            "file_id": file_id,
            "target_column": "label",
            "task_type": "classification",
            "algorithm": "xgboost",
        },
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "completed"


# ---------------------------------------------------------------------------
# Happy path — regression
# ---------------------------------------------------------------------------

def test_train_random_forest_regression(client, auth_token):
    file_id = upload_file(client, auth_token, "regression.csv")
    resp = client.post(
        "/api/training/train",
        json={
            "file_id": file_id,
            "target_column": "value",
            "task_type": "regression",
            "algorithm": "random_forest",
        },
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "completed"
    assert "mse" in data["metrics"]
    assert "rmse" in data["metrics"]
    assert "mae" in data["metrics"]
    assert "r2_score" in data["metrics"]


def test_train_lightgbm_regression(client, auth_token):
    file_id = upload_file(client, auth_token, "regression.csv")
    resp = client.post(
        "/api/training/train",
        json={
            "file_id": file_id,
            "target_column": "value",
            "task_type": "regression",
            "algorithm": "lightgbm",
        },
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "completed"


def test_train_xgboost_regression(client, auth_token):
    file_id = upload_file(client, auth_token, "regression.csv")
    resp = client.post(
        "/api/training/train",
        json={
            "file_id": file_id,
            "target_column": "value",
            "task_type": "regression",
            "algorithm": "xgboost",
        },
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "completed"


# ---------------------------------------------------------------------------
# Custom hyperparameters
# ---------------------------------------------------------------------------

def test_train_custom_hyperparameters(client, auth_token):
    file_id = upload_file(client, auth_token, "classification.csv")
    resp = client.post(
        "/api/training/train",
        json={
            "file_id": file_id,
            "target_column": "label",
            "task_type": "classification",
            "algorithm": "random_forest",
            "hyperparameters": {"n_estimators": 50, "max_depth": 5},
        },
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "completed"


def test_train_invalid_hyperparameter(client, auth_token):
    file_id = upload_file(client, auth_token, "classification.csv")
    resp = client.post(
        "/api/training/train",
        json={
            "file_id": file_id,
            "target_column": "label",
            "task_type": "classification",
            "algorithm": "random_forest",
            "hyperparameters": {"totally_fake_param": 999},
        },
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_train_missing_target_column(client, auth_token):
    file_id = upload_file(client, auth_token, "classification.csv")
    resp = client.post(
        "/api/training/train",
        json={
            "file_id": file_id,
            "target_column": "nonexistent_column",
            "task_type": "classification",
            "algorithm": "random_forest",
        },
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 400
    assert "nonexistent_column" in resp.json()["detail"]


def test_train_non_numeric_features(client, auth_token):
    """has_categorical.csv has a 'dept' string column — should fail validation."""
    file_id = upload_file(client, auth_token, "has_categorical.csv")
    resp = client.post(
        "/api/training/train",
        json={
            "file_id": file_id,
            "target_column": "salary",
            "task_type": "regression",
            "algorithm": "random_forest",
        },
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 400
    assert "Non-numeric" in resp.json()["detail"]


def test_train_invalid_task_type(client, auth_token):
    file_id = upload_file(client, auth_token, "classification.csv")
    resp = client.post(
        "/api/training/train",
        json={
            "file_id": file_id,
            "target_column": "label",
            "task_type": "clustering",
            "algorithm": "random_forest",
        },
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 422


def test_train_invalid_algorithm(client, auth_token):
    file_id = upload_file(client, auth_token, "classification.csv")
    resp = client.post(
        "/api/training/train",
        json={
            "file_id": file_id,
            "target_column": "label",
            "task_type": "classification",
            "algorithm": "neural_network",
        },
        headers=auth_headers(auth_token),
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------

def test_train_file_ownership_enforced(client, auth_token):
    file_id = upload_file(client, auth_token, "classification.csv")

    client.post(
        "/api/auth/register",
        json={"username": "other_trainer", "email": "trainer2@example.com", "password": "password123"},
    )
    other_token = client.post(
        "/api/auth/login",
        json={"username": "other_trainer", "password": "password123"},
    ).json()["access_token"]

    resp = client.post(
        "/api/training/train",
        json={
            "file_id": file_id,
            "target_column": "label",
            "task_type": "classification",
            "algorithm": "random_forest",
        },
        headers=auth_headers(other_token),
    )
    assert resp.status_code == 403


def test_train_unauthenticated(client):
    resp = client.post(
        "/api/training/train",
        json={
            "file_id": "any-id",
            "target_column": "label",
            "task_type": "classification",
            "algorithm": "random_forest",
        },
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Job management
# ---------------------------------------------------------------------------

def test_list_training_jobs(client, auth_token):
    file_id = upload_file(client, auth_token, "classification.csv")
    for _ in range(2):
        client.post(
            "/api/training/train",
            json={
                "file_id": file_id,
                "target_column": "label",
                "task_type": "classification",
                "algorithm": "random_forest",
            },
            headers=auth_headers(auth_token),
        )
    resp = client.get("/api/training/jobs", headers=auth_headers(auth_token))
    assert resp.status_code == 200
    assert len(resp.json()["jobs"]) >= 2


def test_list_jobs_filter_by_file_id(client, auth_token):
    file_id = upload_file(client, auth_token, "classification.csv")
    client.post(
        "/api/training/train",
        json={
            "file_id": file_id,
            "target_column": "label",
            "task_type": "classification",
            "algorithm": "random_forest",
        },
        headers=auth_headers(auth_token),
    )
    resp = client.get(f"/api/training/jobs?file_id={file_id}", headers=auth_headers(auth_token))
    assert resp.status_code == 200
    jobs = resp.json()["jobs"]
    assert all(j["file_id"] == file_id for j in jobs)


def test_get_training_job(client, auth_token):
    file_id = upload_file(client, auth_token, "classification.csv")
    train_resp = client.post(
        "/api/training/train",
        json={
            "file_id": file_id,
            "target_column": "label",
            "task_type": "classification",
            "algorithm": "random_forest",
        },
        headers=auth_headers(auth_token),
    )
    job_id = train_resp.json()["job_id"]

    resp = client.get(f"/api/training/jobs/{job_id}", headers=auth_headers(auth_token))
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id
    assert data["status"] == "completed"
    assert data["metrics"] is not None


def test_get_training_job_other_user_forbidden(client, auth_token):
    file_id = upload_file(client, auth_token, "classification.csv")
    train_resp = client.post(
        "/api/training/train",
        json={
            "file_id": file_id,
            "target_column": "label",
            "task_type": "classification",
            "algorithm": "random_forest",
        },
        headers=auth_headers(auth_token),
    )
    job_id = train_resp.json()["job_id"]

    client.post(
        "/api/auth/register",
        json={"username": "snoop", "email": "snoop@example.com", "password": "password123"},
    )
    other_token = client.post(
        "/api/auth/login",
        json={"username": "snoop", "password": "password123"},
    ).json()["access_token"]

    resp = client.get(f"/api/training/jobs/{job_id}", headers=auth_headers(other_token))
    assert resp.status_code == 403


def test_delete_training_job(client, auth_token):
    file_id = upload_file(client, auth_token, "classification.csv")
    train_resp = client.post(
        "/api/training/train",
        json={
            "file_id": file_id,
            "target_column": "label",
            "task_type": "classification",
            "algorithm": "random_forest",
        },
        headers=auth_headers(auth_token),
    )
    job_id = train_resp.json()["job_id"]
    model_path = train_resp.json()["model_filepath"]

    del_resp = client.delete(f"/api/training/jobs/{job_id}", headers=auth_headers(auth_token))
    assert del_resp.status_code == 200

    # Job is gone from DB
    get_resp = client.get(f"/api/training/jobs/{job_id}", headers=auth_headers(auth_token))
    assert get_resp.status_code == 404

    # Model file is gone from disk
    assert not os.path.exists(model_path)
