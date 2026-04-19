FROM python:3.11-slim

WORKDIR /app

# For health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5001

COPY serve.py .

CMD ["python", "serve.py"]
