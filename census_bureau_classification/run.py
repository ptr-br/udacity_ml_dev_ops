# run.py (top-level)
import sys
from pathlib import Path

import uvicorn

# Ensure src/ is on sys.path so "census" package is importable
sys.path.append(str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    uvicorn.run(
        "census.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # set True for local dev hot-reload
        workers=2,  # increase for prod-ish
    )
