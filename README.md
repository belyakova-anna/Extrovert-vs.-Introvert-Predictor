# 🎭 Introverts vs Extroverts – Personality Classifier

An end-to-end ML pipeline: data preparation, model training, API deployment, and automation with Airflow.

## 📂 Project Structure

```
├── code/
│   ├── datasets/
│   │   ├── data.csv          # Raw source data
│   │   └── prepare.py        # Data preparation script
│   ├── models/
│   │   ├── train.py          # Model training script
│   │   └── main.ipynb        # Jupyter notebook
│   └── deployment/
│       ├── api/
│       │   ├── main.py       # FastAPI server
│       │   └── Dockerfile    # API container
│       ├── app/
│       │   ├── app.py        # Streamlit UI
│       │   └── Dockerfile    # App container
│       └── docker-compose.yml # Run API + App
├── data/
│   ├── raw/
│   │   └── data.csv          # Source data
│   └── processed/
│       ├── train.csv         # Training data (generated)
│       └── test.csv          # Test data (generated)
├── models/
│   └── model.pkl             # Trained model (generated)
├── services/
│   └── airflow/
│       ├── dags/
│       │   └── pipeline.py   # ML pipeline DAG
│       ├── logs/             # Airflow logs
│       └── docker-compose.yml # Airflow setup
└── requirements.txt          # Python dependencies
```

## 🚀 Quick Start

### 1. Data Preparation
```bash
# Prepare train/test splits from raw data
python code/datasets/prepare.py
```

### 2. Model Training
```bash
# Train RandomForest model and save to models/model.pkl
python code/models/train.py
```

### 3. Deploy API + App
```bash
cd code/deployment
docker compose up --build
```
- **API**: http://localhost:8000 (FastAPI)
- **App**: http://localhost:8501 (Streamlit UI)
- **API Docs**: http://localhost:8000/docs

## 🔄 Automated Pipeline (Airflow)

Run the complete ML pipeline with Airflow:

```bash
cd services/airflow
docker compose up --build
```

- **Airflow UI**: http://localhost:8080 (admin/admin)
- **DAG**: `ml_pipeline` runs every 5 minutes
- **Tasks**: `prepare_data` → `train_model` → `deploy_app`

## 🖥️ Local Development

### Setup Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate
# Activate (Windows PowerShell)
. .venv/Scripts/Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Manual Execution
```bash
# 1. Prepare data
python code/datasets/prepare.py

# 2. Train model
python code/models/train.py

# 3. Run API
uvicorn code.deployment.api.main:app --host 0.0.0.0 --port 8000

# 4. Run Streamlit (new terminal)
export API_BASE_URL=http://localhost:8000  # Linux/macOS
# $env:API_BASE_URL='http://localhost:8000'  # Windows PowerShell
streamlit run code/deployment/app/app.py
```

## 📡 API Endpoints

### Health Check
```bash
GET /health
# Response: { "status": "ok" }
```

### Prediction (Typed Schema)
```bash
POST /predict
Content-Type: application/json

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

**Response**: `{ "predictions": [0] }` where 0=Introvert, 1=Extrovert

### Prediction (Legacy Schema)
```bash
POST /predict
Content-Type: application/json

{
  "features": [2.0, 6.0, 3.0, 10.0, 5.0, "Yes", "No"]
}
```

## 🔢 Inputs
- Time spent alone: 0–10 plus “10+” (maps to 11)
- Friends circle size: 0–14 plus “14+” (maps to 15)
- Social event attendance: 0–10
- Social media posting frequency: 0–10
- Going outside: days/week 0–7
- Stage fright / Drained after socializing: Yes/No
