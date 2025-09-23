FROM apache/airflow:2.8.4-python3.10

# Сначала зависимости Python
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

# Теперь пакеты Ubuntu
USER root
RUN apt-get update && apt-get install -y docker.io docker-compose-plugin && rm -rf /var/lib/apt/lists/*

# Вернуться обратно под пользователем airflow
USER airflow
