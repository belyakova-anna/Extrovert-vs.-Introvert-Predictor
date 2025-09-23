"""
Airflow DAG: ML pipeline automation

Tasks:
- prepare_data: runs code/datasets/prepare.py
- train_model: runs code/models/train.py
- deploy_app: runs docker compose up --build -d in code/deployment

Schedule: every 5 minutes (cron: "*/5 * * * *")
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import os

from airflow import DAG
from airflow.operators.bash import BashOperator


# Resolve repository root:
# 1) Prefer PROJECT_ROOT env (set by docker-compose), else
# 2) Fallback: infer from file path when running from repo directly
_project_root_env = os.environ.get("PROJECT_ROOT")
if _project_root_env:
    REPO_ROOT = Path(_project_root_env).resolve()
else:
    # File path when repo structure is used directly:
    # <repo>/services/airflow/dags/pipeline.py
    # parents[0]=.../dags, [1]=.../airflow, [2]=.../services, [3]=<repo>
    REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ml_pipeline",
    description="Prepare data -> Train model -> Deploy app",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule_interval="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
) as dag:
    # Commands
    prepare_cmd = f'cd "{REPO_ROOT}" && python code/datasets/prepare.py'
    train_cmd = f'cd "{REPO_ROOT}" && python code/models/train.py'
    deploy_cmd = f'cd "{REPO_ROOT}/code/deployment" && docker compose up --build -d'

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=prepare_cmd,
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=train_cmd,
    )

    deploy_app = BashOperator(
        task_id="deploy_app",
        bash_command=deploy_cmd,
    )

    # Chain: prepare_data -> train_model -> deploy_app
    prepare_data >> train_model >> deploy_app
