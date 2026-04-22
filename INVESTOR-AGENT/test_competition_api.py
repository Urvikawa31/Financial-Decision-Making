import requests
import json

def test_api():
    url = "http://localhost:62237/trading_action/"
    
    # Sample A (TSLA)
    payload = {
        "date": "2025-01-15",
        "price": {"TSLA": 250.50},
        "news": {"TSLA": ["Tesla announces new production milestone"]},
        "symbol": ["TSLA"],
        "momentum": {"TSLA": "bullish"},
        "10k": {"TSLA": ["[SEC 10-K Filing - 2025-01-15]\nSummary..."]},
        "10q": {"TSLA": ["[SEC 10-Q Filing - 2025-01-15]\nSummary..."]},
        "history_price": {"TSLA": [
            {"date": "2025-01-12", "price": 249.80},
            {"date": "2025-01-13", "price": 250.50},
            {"date": "2025-01-14", "price": 250.30}
        ]}
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload, timeout=180) # 3 min timeout as per spec
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
