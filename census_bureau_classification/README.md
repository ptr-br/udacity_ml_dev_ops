# Getting Started 
This project builds and serves a Census Income classification model end‑to‑end. It includes a reproducible preprocessing pipeline, model training, evaluation with overall and slice metrics, and a FastAPI service for secure REST inference. It’s ready to run locally with uv/uvicorn and can be deployed to a managed platform with minimal changes.

### Installation (uv)
For running with conda please createe a venv, activate it and use `python` insetead of `uv`

`uv run`

```bash 
uv sync --index https://pypi.org/simple/
```

### Clean the data

```bash
uv run src/census/data/clean_data.py src/census/data/census.csv
```

### Train the model
```bash
uv run src/census/model/train_model.py 
```

### Run the API Locally 
```bash
uv run run.py
```

### Run tests (Model + API)
```bash
uv run pytest .
``` 

## Interact with Render live API
```bash
uv run call_live_api.py
```
> ⚠️ Live API calls are temporary and will be disabled soon to avoid cloud costs. Please refer to local setup.
