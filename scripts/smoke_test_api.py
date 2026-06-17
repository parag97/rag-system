import requests


response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What is Amazon?"
    },
)

print(response.status_code)
print(response.json())