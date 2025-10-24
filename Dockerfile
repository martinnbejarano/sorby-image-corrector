# ===== Runtime base =====
FROM python:3.12-slim

# Evita .pyc y fuerza logs sin buffer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    APP_HOME=/app

WORKDIR ${APP_HOME}

# Dependencias del sistema necesarias para opencv-python
# (libgl1 y libglib2.0-0 evitan errores de OpenCV al importar)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# (Opcional) Si tu requirements.txt existe, lo usamos para cachear
# Si no existe, el RUN de abajo instala los paquetes explícitos.
COPY requirements.txt ./ 2>/dev/null || true

# Instala dependencias de Python:
# - Si hay requirements.txt lo usa,
# - Si no, instala las versiones que listaste.
RUN python -m pip install --upgrade --no-cache-dir pip wheel setuptools && \
    ( test -f requirements.txt && \
      pip install --no-cache-dir -r requirements.txt \
      || pip install --no-cache-dir \
           fastapi==0.115.0 \
           uvicorn==0.30.6 \
           opencv-python==4.12.0.88 \
           numpy==2.1.2 \
           httpx==0.27.2 \
           python-multipart==0.0.9 )

# Copiá el código de la app (asegurate que main.py esté en la raíz o ajustá CMD)
COPY . .

# (Opcional) Crear usuario no-root
RUN useradd -m appuser && chown -R appuser:appuser ${APP_HOME}
USER appuser

EXPOSE ${PORT}

# Ejecuta Uvicorn en "modo producción" (4 workers)
# Si preferís hot-reload en desarrollo, reemplazá por: ["uvicorn","main:app","--reload","--host","0.0.0.0","--port","8000"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
