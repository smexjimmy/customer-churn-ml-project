FROM python:3.12-slim

LABEL org.opencontainers.image.title="Churn Prediction API"
LABEL org.opencontainers.image.description="FastAPI service for customer churn prediction"

WORKDIR /app

# Install dependencies first (cached layer)
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY params.yaml .

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
