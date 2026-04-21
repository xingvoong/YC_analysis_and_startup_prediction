FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy serving code
COPY . /app/serving/

# Copy model artifacts
COPY ../../yc_startup_validator_app/model /app/yc_startup_validator_app/model

WORKDIR /app/serving
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
