import os
import pytest


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def uploaded_file_id(client, auth_token):
    """Upload a sample CSV and return its file_id."""
    sample_csv = os.path.join(FIXTURES_DIR, "sample.csv")
    with open(sample_csv, "rb") as f:
        response = client.post(
            "/api/files/upload",
            files={"file": ("sample.csv", f, "text/csv")},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
    assert response.status_code == 200
    return response.json()["file_id"]


def auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def test_get_data_summary(client, auth_token, uploaded_file_id):
    response = client.get(
        f"/api/data/summary/{uploaded_file_id}",
        headers=auth_headers(auth_token),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["file_id"] == uploaded_file_id
    assert "summary" in data


def test_get_summary_unauthorized(client, auth_token, uploaded_file_id):
    response = client.get(f"/api/data/summary/{uploaded_file_id}")
    assert response.status_code == 403


def test_get_summary_other_user_file(client, db_setup, uploaded_file_id):
    """A different user should not be able to access another user's file."""
    # Register a second user
    client.post(
        "/api/auth/register",
        json={"username": "other", "email": "other@example.com", "password": "password123"},
    )
    login_resp = client.post(
        "/api/auth/login",
        json={"username": "other", "password": "password123"},
    )
    other_token = login_resp.json()["access_token"]

    response = client.get(
        f"/api/data/summary/{uploaded_file_id}",
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert response.status_code == 403


def test_preprocess_missing_values(client, auth_token, uploaded_file_id):
    response = client.post(
        f"/api/data/preprocess/{uploaded_file_id}",
        json={
            "handle_missing": True,
            "missing_config": {"strategy": "remove_column", "missing_threshold": 0.5},
        },
        headers=auth_headers(auth_token),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "preprocessing_steps" in data


def test_handle_missing_values_endpoint(client, auth_token, uploaded_file_id):
    response = client.post(
        f"/api/data/missing-values/{uploaded_file_id}",
        params={"strategy": "remove_column", "missing_threshold": 0.8},
        headers=auth_headers(auth_token),
    )
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_handle_outliers_endpoint(client, auth_token, uploaded_file_id):
    response = client.post(
        f"/api/data/outliers/{uploaded_file_id}",
        params={"method": "iqr", "threshold": 1.5},
        headers=auth_headers(auth_token),
    )
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_encode_categorical_endpoint(client, auth_token, uploaded_file_id):
    response = client.post(
        f"/api/data/encode/{uploaded_file_id}",
        params={"method": "label"},
        headers=auth_headers(auth_token),
    )
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_scale_features_endpoint(client, auth_token, uploaded_file_id):
    response = client.post(
        f"/api/data/scale/{uploaded_file_id}",
        params={"method": "standard"},
        headers=auth_headers(auth_token),
    )
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_remove_duplicates_endpoint(client, auth_token, uploaded_file_id):
    response = client.delete(
        f"/api/data/duplicates/{uploaded_file_id}",
        headers=auth_headers(auth_token),
    )
    assert response.status_code == 200
    assert response.json()["success"] is True
