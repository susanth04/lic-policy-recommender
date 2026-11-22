# ==============================================================================
# Dockerfile for LIC Policy Recommender (FastAPI Backend + HTML Dashboard)
# ==============================================================================
FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing .pyc files and buffer output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy and install core runtime requirements (gbr_only mode: scikit-learn, numpy, fastapi, uvicorn, etc.)
COPY model_out/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir python-dotenv sqlalchemy

# Copy application directories and env template
COPY app /app/app
COPY model_out /app/model_out
COPY data /app/data
COPY tools /app/tools
COPY .env* /app/

# Default port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
