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
    fastapi \
    uvicorn[standard] \
    pandas \
    scikit-learn \
    pydantic

# Copy project (build context is repo root)
COPY . /app

EXPOSE 8000

# Run from the API directory so imports work without treating 'code' as a package
WORKDIR /app/code/deployment/api
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


