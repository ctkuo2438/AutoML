def test_register_user(client):
    response = client.post(
        "/api/auth/register",
        json={"username": "alice", "email": "alice@example.com", "password": "secret123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "alice"
    assert data["email"] == "alice@example.com"
    assert "id" in data
    assert "hashed_password" not in data


def test_register_duplicate_username(client):
    payload = {"username": "bob", "email": "bob@example.com", "password": "password123"}
    client.post("/api/auth/register", json=payload)
    response = client.post(
        "/api/auth/register",
        json={"username": "bob", "email": "bob2@example.com", "password": "password123"},
    )
    assert response.status_code == 400
    assert "Username already registered" in response.json()["detail"]


def test_register_duplicate_email(client):
    client.post(
        "/api/auth/register",
        json={"username": "carol", "email": "shared@example.com", "password": "password123"},
    )
    response = client.post(
        "/api/auth/register",
        json={"username": "carol2", "email": "shared@example.com", "password": "password123"},
    )
    assert response.status_code == 400
    assert "Email already registered" in response.json()["detail"]


def test_register_invalid_email(client):
    response = client.post(
        "/api/auth/register",
        json={"username": "baduser", "email": "not-an-email", "password": "password123"},
    )
    assert response.status_code == 422


def test_register_password_too_short(client):
    response = client.post(
        "/api/auth/register",
        json={"username": "baduser", "email": "valid@example.com", "password": "short"},
    )
    assert response.status_code == 422


def test_login_valid_credentials(client):
    client.post(
        "/api/auth/register",
        json={"username": "charlie", "email": "charlie@example.com", "password": "mypassword"},
    )
    response = client.post(
        "/api/auth/login",
        json={"username": "charlie", "password": "mypassword"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_wrong_password(client):
    client.post(
        "/api/auth/register",
        json={"username": "dave", "email": "dave@example.com", "password": "correctpassword"},
    )
    response = client.post(
        "/api/auth/login",
        json={"username": "dave", "password": "wrong"},
    )
    assert response.status_code == 401


def test_login_nonexistent_user(client):
    response = client.post(
        "/api/auth/login",
        json={"username": "nobody", "password": "pass"},
    )
    assert response.status_code == 401


def test_get_current_user(client, registered_user, auth_token):
    response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert response.status_code == 200
    assert response.json()["username"] == registered_user["username"]


def test_get_current_user_invalid_token(client):
    response = client.get(
        "/api/auth/me",
        headers={"Authorization": "Bearer invalidtoken"},
    )
    assert response.status_code == 401
