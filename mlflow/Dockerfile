FROM python:3.12-slim

LABEL org.opencontainers.image.title="MLflow Tracking Server"
LABEL org.opencontainers.image.description="MLflow with PostgreSQL backend and MinIO artifact store"

# setuptools provides pkg_resources — required by mlflow deps on Python 3.12-slim
# Install it first before mlflow
RUN pip install --no-cache-dir "setuptools<82" wheel && \
    pip install --no-cache-dir \
        mlflow==3.14.0 \
        boto3==1.34.144 \
        psycopg2-binary==2.9.9

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
