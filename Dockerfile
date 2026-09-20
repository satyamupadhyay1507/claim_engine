FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose ports: 7860 (Hugging Face Spaces default), 8000 (FastAPI), 8501 (Streamlit)
EXPOSE 7860 8000 8501

ENV PORT=7860

CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT}"]
