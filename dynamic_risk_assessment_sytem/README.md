# Dynamic Risk Assessment Projcet

We use `uv` for packet management.
To get started please setup the envirment using `uv sync`

To run the full pipeline, first start the Flask server via `uv run app.py` and then execute: 
```bash
uv run ./fullprocess.py
```

# Available Scripts

## Data Ingestion 
```bash
uv run ingestion.py
```

## Model Training  

```bash
uv run training.py
```

## Reporting & Scoring
```bash
uv run reporting.py && uv run scoring.py
```

## Deploy model
```bash
uv run deployment.py
```

## Run Diagnostics
```bash
uv run deployment.py
```

## Query API
```bash
 uv run apicalls.py 
```
> Note: Flask server needs to be running!

### Linting and Formatting
```bash
uvx ruff check . --fix && uvx ruff format
```

