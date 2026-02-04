"""
Run the FastAPI RAG demo locally. Usage: python scripts/serve.py
Serves at http://127.0.0.1:8000
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
