FROM python:3.11-slim

WORKDIR /app

# Required for MLFlow and DVC
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ git curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Initialize DVC without git (pipeline runner mode)
RUN dvc init --no-scm

COPY params.yaml .
COPY dvc.yaml .
COPY src/ src/
COPY data/ data/
COPY tests/ tests/
RUN mkdir -p models metrics data/ingested data/transformed

EXPOSE 8001

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8001"]
