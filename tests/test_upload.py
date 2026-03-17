import os
import pytest


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def test_upload_csv_authenticated(client, auth_token):
    sample_csv = os.path.join(FIXTURES_DIR, "sample.csv")
    with open(sample_csv, "rb") as f:
        response = client.post(
            "/api/files/upload",
            files={"file": ("sample.csv", f, "text/csv")},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
    assert response.status_code == 200
    data = response.json()
    assert "file_id" in data
    assert data["filename"] == "sample.csv"


def test_upload_csv_unauthenticated(client):
    sample_csv = os.path.join(FIXTURES_DIR, "sample.csv")
    with open(sample_csv, "rb") as f:
        response = client.post(
            "/api/files/upload",
            files={"file": ("sample.csv", f, "text/csv")},
        )
    assert response.status_code == 403


def test_upload_non_csv_rejected(client, auth_token):
    response = client.post(
        "/api/files/upload",
        files={"file": ("data.txt", b"not a csv", "text/plain")},
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert response.status_code in (400, 422)
