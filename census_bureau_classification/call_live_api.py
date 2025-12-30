import requests


def main():
    base_url = "https://udacity-ml-dev-ops.onrender.com"
    url = f"{base_url}/infer"

    payload = {
        "age": 50,
        "workclass": "Private",
        "fnlgt": 234721,
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Separated",
        "occupation": "Exec-managerial",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States",
    }

    resp = requests.post(url, json=payload, timeout=15)
    print("Status:", resp.status_code)
    print("Content-Type:", resp.headers.get("content-type"))
    try:
        print("JSON:", resp.json())
    except Exception:
        print("Raw response:", resp.text)


if __name__ == "__main__":
    main()
