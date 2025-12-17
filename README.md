# 🛡️ MLOps Pipeline: Insurance Cross-Selling API

Este repositorio contiene un proyecto de Machine Learning con un pipeline de CI/CD completamente funcional. El objetivo es predecir si un cliente asegurado está interesado en comprar un seguro de vehículo adicional (Cross-Selling).

La arquitectura sigue el patrón MLOps: Código en Git, Datos en S3, Artefactos en ECR, y Automatización con GitHub Actions.

## ⚙️ Arquitectura del Sistema

| Componente | Uso Principal | Tecnología |
| :--- | :--- | :--- |
| **Código** | Almacenamiento y Versionamiento del código fuente. | GitHub |
| **Datos** | Almacenamiento centralizado e inmutable de `train.csv`, `test.csv`, `production.csv`. | AWS S3 (`insurance-mlops-data-10056927`) |
| **Pipeline** | Orquestación de entrenamiento, versionado e inyección de ID del modelo. | GitHub Actions (CI/CD) |
| **API** | Servicio de inferencia para predicciones en tiempo real. | FastAPI, Uvicorn |
| **Artefacto** | Registro de modelos, métricas y parámetros (local en CI/CD). | MLflow |
| **Contenedor** | Imagen Docker de la API lista para despliegue. | AWS ECR (`insurance-api`) |

---

## 📝 Guía de Inicio Rápido

Sigue estos pasos para re-entrenar el modelo, construir la API y desplegar la imagen en AWS.

### 1. Pre-requisitos

* Docker Desktop (corriendo).
* AWS CLI configurada y autenticada.
* Credenciales AWS con permisos de S3 Read (para datos) y ECR Push (para imágenes).

### 2. Configuración y Limpieza Local

Asegúrate de tener un entorno Python con todas las dependencias instaladas y limpia los logs de la ejecución anterior:

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Limpiar logs y DB locales (Para evitar errores de rutas)
rm -f latest_run_id.txt mlflow.db
rm -rf mlruns
```

3. Ejecutar el Pipeline de CI/CD (Entrenamiento y Empaquetado)
Cualquier cambio empujado a la rama main dispara automáticamente el proceso de entrenamiento y despliegue:

Haz un cambio en el código (ej. ajusta un parámetro en src/training/train.py).

Sube los cambios a main:

```bash

git add .
git commit -m "feat: Nuevo re-entrenamiento con cambio de hiperparámetro"
git push origin main
```
El Production Pipeline se ejecutará en GitHub Actions, completando las siguientes tareas:

Descarga de train.csv desde S3.

Entrenamiento del modelo.

Inyección del nuevo RUN_ID en src/app/main.py.

Construcción de la imagen Docker.

Subida de la imagen (:latest y :hash) a ECR.

4. Prueba Local del Artefacto de Producción
Una vez que GitHub Actions esté en ✅ verde, puedes probar la imagen de ECR en tu máquina:

A. Autenticar Docker
```bash

# Reemplaza 1234567890 con tu ID de cuenta si esta guía se usa en otro lugar
AWS_ACCOUNT_ID=1234567890
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
B. Descargar y Ejecutar la API
Bash

IMAGE_URI=$AWS_ACCOUNT_[ID.dkr.ecr.us-east-1.amazonaws.com/insurance-api:latest](https://ID.dkr.ecr.us-east-1.amazonaws.com/insurance-api:latest)

# Descargar la versión más reciente
docker pull $IMAGE_URI

# Ejecutar el contenedor en segundo plano en el puerto 8001
docker run -d -p 8001:8000 --name insurance-prod-test $IMAGE_URI
C. Probar la Predicción (Health Check y cURL)
Accede a http://localhost:8001/docs para ver la documentación de FastAPI o usa curl:
```
```bash

curl -X 'POST' \
  'http://localhost:8001/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "Gender": "Male",
  "Age": 44,
  "HasDrivingLicense": 1,
  "RegionID": 28.0,
  "Switch": 0.0,
  "VehicleAge": "> 2 Years",
  "PastAccident": "Yes",
  "AnnualPremium": "£40454.0",
  "SalesChannelID": 26.0,
  "DaysSinceCreated": 217
}'
```
🧹 Limpieza de Entorno Local (Importante)
Para liberar los recursos de tu laptop (sin borrar los recursos de AWS):

```bash

# 1. Eliminar el contenedor que está corriendo
docker rm -f insurance-prod-test

# 2. Eliminar logs, bases de datos y archivos temporales
rm -f latest_run_id.txt mlflow.db production_log.json
rm -rf mlruns
rm -f data/train.csv data/test.csv data/production.csv
