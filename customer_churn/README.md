# Predict Customer Churn

Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity


## Quick Start

### Prerequisites

- Python 3.8 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip

### Setup with uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh


# Install all dependencies (creates .venv automatically)
uv sync

# Run the main pipeline
uv run python churn_library.py

# Runu testscustomer_churn/README.md
uv rn python churn_script_logging_and_tests.py

# Check code quality
uv run pylint churn_library.py
```


**Test results:**
- All test outputs logged to `./logs/churn_library.log`
- SUCCESS/ERROR status for each function
- Detailed assertions and validations

### Code Quality Checks

```bash
# Run pylint on main library
uv run pylint churn_library.py

# Run pylint on test suite
uv run pylint churn_script_logging_and_tests.py

# Format code with autopep8
uv run autopep8 --in-place --aggressive --aggressive churn_library.py
```

## Author

ptr-br  
Date: December 10, 2025

This project is part of the Udacity ML DevOps Engineer Nanodegree program.



