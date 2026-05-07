FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (layer cache friendly)
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY . .

EXPOSE 8000
CMD ["python3", "-m", "api.main"]
