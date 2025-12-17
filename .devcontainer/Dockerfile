FROM python:3.9-slim

# Instalamos git (necesario para DVC), curl y unzip (para AWS CLI si fuera necesario)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Configuración de trabajo
WORKDIR /workspace

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar AWS CLI v2 (Opcional si montamos el volumen, pero recomendado tenerlo dentro)
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip