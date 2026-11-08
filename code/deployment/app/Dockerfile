FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (optional minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    streamlit \
    requests \
    pandas \
    scikit-learn

# Copy project (build context is repo root)
COPY . /app

ENV API_BASE_URL="http://api:8000"

EXPOSE 8501

CMD ["streamlit", "run", "code/deployment/app/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]


