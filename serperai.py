import requests

SERPAPI_KEY = "cf89a23338e13b7a6c028ff2aa6e51ffd6c283ad0e0fe2226f8e2ae8e77c17a3"

def test_serpapi():
    url = "https://serpapi.com/search"
    params = {
        "q": "Artificial intelligence Wikipedia",
        "api_key": SERPAPI_KEY,
        "engine": "google",
        "num": 5
    }
    response = requests.get(url, params=params)
    print("Status code:", response.status_code)
    print("Response:", response.text)

test_serpapi()
