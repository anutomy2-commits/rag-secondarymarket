FROM python:3.12-slim

WORKDIR /app

# build-essential: some ML deps (e.g. chromadb's hnswlib) fall back to
# compiling from source if no prebuilt wheel matches this exact platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
