import pandas as pd
import mlflow.xgboost
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from src.app.schemas import InsuranceInput, PredictionOutput
from src.training.preprocess import clean_data, feature_engineering

# --- CONFIGURACIÓN ---
# 1. Configurar la URI (¡ESTO FALTABA!) 🆕
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 2. Tu Run ID Ganador 
RUN_ID = "1eb30ea8c1f84e84b7934b59f451855c"
MODEL_URI = f"runs:/{RUN_ID}/model"
# Variables globales
model = None
BEST_THRESHOLD = 0.67 

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"🔄 Descargando modelo desde: {MODEL_URI}")
    try:
        # Ahora sí buscará en la base de datos SQLite
        model = mlflow.xgboost.load_model(MODEL_URI)
        print("✅ Modelo cargado correctamente.")
    except Exception as e:
        print(f"❌ ERROR CRÍTICO: {e}")
        # Tip de depuración: Imprimir dónde está buscando
        import os
        print(f"Directorio actual: {os.getcwd()}")
    yield
    model = None

app = FastAPI(title="Insurance API", version="1.0", lifespan=lifespan)

@app.get("/")
def health_check():
    """Endpoint para verificar si la API está viva."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: InsuranceInput):
    if not model:
        raise HTTPException(status_code=503, detail="El modelo no está listo.")
    
    try:
        # 1. Convertir JSON a DataFrame
        data_dict = input_data.model_dump()
        df = pd.DataFrame([data_dict])
        
        # 2. Preprocesamiento (REUTILIZAMOS EL CÓDIGO DE ENTRENAMIENTO)
        # Esto garantiza que 'train' y 'predict' hablen el mismo idioma
        df_clean = clean_data(df)
        df_processed = feature_engineering(df_clean)
        
        # 3. Inferencia
        probs = model.predict_proba(df_processed)[:, 1]
        prob_value = float(probs[0])
        
        # 4. Decisión con Umbral Optimizado
        prediction = 1 if prob_value >= BEST_THRESHOLD else 0
        
        return {
            "prediction": prediction,
            "probability": round(prob_value, 4),
            "threshold_used": BEST_THRESHOLD,
            "model_version": RUN_ID
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")