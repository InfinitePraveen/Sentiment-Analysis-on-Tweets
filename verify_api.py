from app import app

client = app.test_client()
response = client.post("/predict", json={"tweet": "I love this app"})
print("status:", response.status_code)
print("json:", response.get_json())
