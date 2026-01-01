# apicalls.py
import requests
import json
import os

URL = "http://127.0.0.1:8000"

if __name__ == "__main__":
# Specify a URL that resolves to your workspace


    # Load output path from config
    with open("config.json", "r") as f:
        config = json.load(f)

    output_model_path = os.path.join(config["output_model_path"])
    os.makedirs(output_model_path, exist_ok=True)
    out_file = os.path.join(output_model_path, "apireturns.txt")

    # Call each API endpoint and store the responses
    base = URL.rstrip("/")  # ensure no trailing slash
    response1 = requests.post(f"{base}/prediction",
                            json={"filepath": os.path.join(config["test_data_path"], "testdata.csv")})
    response2 = requests.get(f"{base}/scoring")
    response3 = requests.get(f"{base}/summarystats")
    response4 = requests.get(f"{base}/diagnostics")
    responses = {
        "prediction": response1.json(),
        "scoring": response2.json(),
        "summarystats": response3.json(),
        "diagnostics": response4.json()
    }

    with open(out_file, "w") as f:
        f.write(json.dumps(responses, indent=2))

    print(f"API returns saved to {out_file}")