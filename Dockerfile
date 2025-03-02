# Builder Stage
FROM python:3.11-slim AS builder

# Install System Dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    unzip \
    build-essential \
    libpq-dev \
    gcc \
    python3-dev \
    postgresql-client \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libnspr4 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    xdg-utils \
    libglib2.0-0 \
    libx11-6 \
    libmupdf-dev \  
    && rm -rf /var/lib/apt/lists/*

# Install Chromium and ChromeDriver
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Poetry Installation
RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false

# Project Dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry install --with dev --no-root

# Additional Python Packages
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install sentence-transformers && \
    pip install faiss-cpu==1.9.0.post1 && \
    pip install \
        apache-airflow==2.7.3 \
        apache-airflow-providers-celery==3.3.1 \
        apache-airflow-providers-postgres==5.6.0 \
        apache-airflow-providers-redis==3.3.1 \
        apache-airflow-providers-http==4.1.0 \
        apache-airflow-providers-common-sql==1.10.0 \
        croniter==2.0.1 \
        cryptography==42.0.0

# Pre-download the SentenceTransformer model
RUN mkdir -p /app/models/sentence-transformers/all-MiniLM-L6-v2 && \
    # First try to download using the library with SSL verification disabled
    (python -c "import os; os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'; os.environ['TRANSFORMERS_CACHE'] = '/app/models'; os.environ['HF_HOME'] = '/app/models'; from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2')" || \
    # If that fails, download files directly
    (echo "Manual model download as fallback" && \
    cd /app/models/sentence-transformers/all-MiniLM-L6-v2 && \
    wget --no-check-certificate -O model.safetensors https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors && \
    wget --no-check-certificate -O config.json https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json && \
    wget --no-check-certificate -O tokenizer_config.json https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json && \
    wget --no-check-certificate -O tokenizer.json https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json && \
    wget --no-check-certificate -O special_tokens_map.json https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/special_tokens_map.json && \
    wget --no-check-certificate -O modules.json https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json && \
    wget --no-check-certificate -O vocab.txt https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt))

# Final Stage
FROM python:3.11-slim

# Install System Dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    libomp-dev \
    curl \
    wget \
    unzip \
    redis-tools \
    netcat-openbsd \
    chromium \
    chromium-driver \
    libglib2.0-0 \
    libnss3 \
    libx11-6 \
    libmupdf-dev \ 
    && rm -rf /var/lib/apt/lists/*

# User and Group Setup with explicit IDs
RUN groupadd -g 125 appgroup && \
    useradd -u 1001 -g appgroup -s /bin/bash -m appuser

# Create and set permissions for Chrome directories
RUN mkdir -p /tmp/chrome-data /tmp/chrome-profile /var/run/chrome && \
    chown -R 1001:125 /tmp/chrome-data /tmp/chrome-profile /var/run/chrome && \
    chmod -R 1777 /tmp/chrome-data /tmp/chrome-profile /var/run/chrome

# Chrome sandbox setup
RUN chown root:root /usr/bin/chromium && \
    chmod 4755 /usr/bin/chromium

# Directory Structure with Updated Permissions
RUN mkdir -p \
    /app/ai_services_api/services/search/models \
    /app/logs \
    /app/cache \
    /app/models \
    /opt/airflow/logs \
    /opt/airflow/dags \
    /opt/airflow/plugins \
    /opt/airflow/data \
    /app/scripts \
    /app/tests && \
    # Enhanced permissions for FAISS index directory
    chmod -R 777 /app/ai_services_api/services/search/models && \
    chmod -R 777 /app/ai_services_api/services/search && \
    chmod -R 777 /app/models && \
    # General permissions for other directories
    chown -R 1001:125 /app /opt/airflow && \
    chmod -R 775 /app /opt/airflow

# Working Directory
WORKDIR /app

# Copy Dependencies
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/models /app/models

# Application Files with updated permissions
COPY --chown=1001:125 . .
RUN chmod +x /app/scripts/init-script.sh && \
    # Ensure FAISS directory permissions persist after copy
    chmod -R 777 /app/ai_services_api/services/search/models && \
    chmod -R 777 /app/models

# Set Chrome flags
ENV CHROME_FLAGS="--headless=new --no-sandbox --disable-gpu --disable-dev-shm-usage --disable-crashpad --disable-crash-reporter --no-first-run --test-type --disable-software-rasterizer --disable-default-apps --disable-setuid-sandbox --remote-debugging-port=9222"

# Environment Variables
ENV TRANSFORMERS_CACHE=/app/models \
    HF_HOME=/app/models \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    AIRFLOW_HOME=/opt/airflow \
    PYTHONPATH=/app \
    TESTING=false \
    CHROME_BIN=/usr/bin/chromium \
    CHROMEDRIVER_PATH=/usr/bin/chromedriver \
    CHROME_TMPDIR=/tmp/chrome-data \
    CHROME_PROFILE_DIR=/tmp/chrome-profile

# Health Check
HEALTHCHECK --interval=30s \
            --timeout=10s \
            --start-period=60s \
            --retries=3 \
            CMD curl -f http://localhost:8000/health || exit 1

# User Switch
USER 1001:125
ENV DISPLAY=:99

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]