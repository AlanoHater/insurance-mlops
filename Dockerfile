# 1. Imagen base ligera de Python
FROM python:3.9-slim

# 2. Evitar que Python genere archivos .pyc y buffer de salida
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Directorio de trabajo dentro del contenedor
WORKDIR /app

# 4. Instalar dependencias del sistema (necesarias para XGBoost/Pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 5. Copiar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copiar el código fuente
COPY src/ src/

# 7. Copiar la base de datos de experimentos (Importante para cargar el modelo)
# En un entorno real (AWS), esto se sustituye por conexión a S3/RDS
COPY mlflow.db .
COPY mlruns/ mlruns/

# 8. Exponer el puerto
EXPOSE 8000

# 9. Comando de arranque
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]