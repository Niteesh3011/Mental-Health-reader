"""
Launcher script — run from the project root:

    python main.py                        (direct)
    uvicorn app.main:app --reload         (uvicorn CLI)
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
