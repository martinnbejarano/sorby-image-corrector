FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    APP_HOME=/app

WORKDIR ${APP_HOME}

# Dependencias mínimas para OpenCV (si usás opencv-python "no headless")
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# 1) Copiá solo requirements para aprovechar caché
COPY requirements.txt .

# 2) Instalá deps Python
RUN python -m pip install --upgrade --no-cache-dir pip wheel setuptools \
 && pip install --no-cache-dir -r requirements.txt

# 3) Copiá el resto del código
COPY . .

# Usuario no-root (opcional)
RUN useradd -m appuser && chown -R appuser:appuser ${APP_HOME}
USER appuser

EXPOSE ${PORT}

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
