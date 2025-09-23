# 🎭 Introverts vs Extroverts – Personality Classifier

An end-to-end demo: train a model to classify personality (Introvert/Extrovert), serve it via FastAPI, and use a Streamlit UI.

## 📂 Structure
- `code/models/train.py` – training script (writes `models/model.pkl`)
- `code/deployment/api/main.py` – FastAPI (`/health`, `/predict`)
- `code/deployment/app/app.py` – Streamlit app
- `code/deployment/docker-compose.yml` – run API + App
- `code/datasets/data.csv` – sample data
- `models/model.pkl` – trained model (generated)

## 🖥️ Local setup (without Docker)
1) Create venv and install deps
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
. .venv/Scripts/Activate.ps1
pip install -U pip fastapi "uvicorn[standard]" pandas scikit-learn pydantic streamlit requests
```
2) Train model (creates `models/model.pkl`)
```bash
python code/models/train.py
```
3) Run API
```bash
uvicorn code.deployment.api.main:app --host 0.0.0.0 --port 8000
```
4) Run Streamlit (new terminal)
```bash
# Linux/macOS
export API_BASE_URL=http://localhost:8000
# Windows PowerShell
$env:API_BASE_URL='http://localhost:8000'
streamlit run code/deployment/app/app.py
```
Open http://localhost:8501

## 🐳 Docker (recommended)
From `code/deployment`:
```bash
docker compose up --build
```
- API: http://localhost:8000
- App: http://localhost:8501

## 📡 API
- `GET /health` → `{ "status": "ok" }`
- `POST /predict` (typed schema)
```json
{
  "samples": {
    "Time_spent_Alone": 2.0,
    "Social_event_attendance": 6.0,
    "Going_outside": 3.0,
    "Friends_circle_size": 10.0,
    "Post_frequency": 5.0,
    "Stage_fear": "Yes",
    "Drained_after_socializing": "No"
  }
}
```
Returns `{ "predictions": [0] }` where 0=Introvert, 1=Extrovert.

## 🔢 Inputs
- Time spent alone: 0–10 plus “10+” (maps to 11)
- Friends circle size: 0–14 plus “14+” (maps to 15)
- Social event attendance: 0–10
- Social media posting frequency: 0–10
- Going outside: days/week 0–7
- Stage fright / Drained after socializing: Yes/No
