import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from src.training.preprocess import load_data, clean_data, feature_engineering

# Configuración MLflow (Local en Docker)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("insurance-cross-selling-prod")

def main():
    # ⚠️ CAMBIO IMPORTANTE: agregamos 'as run' para capturar el objeto
    with mlflow.start_run() as run:
        print("🚀 [1/5] Iniciando Ingesta y ETL...")
        df = load_data("data/train.csv")
        df = clean_data(df)
        df = feature_engineering(df)
        
        # Validar Target
        TARGET = 'Result'
        if TARGET not in df.columns:
            raise ValueError(f"CRÍTICO: Columna '{TARGET}' no encontrada.")

        # Selección de Features
        X = df.drop(columns=[TARGET], errors='ignore')
        y = df[TARGET]
        
        # Split Estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("⚖️ [2/5] Calculando Balanceo de Clases...")
        count_neg = (y_train == 0).sum()
        count_pos = (y_train == 1).sum()
        scale_weight = count_neg / count_pos
        print(f"   -> Scale Pos Weight: {scale_weight:.2f}")
        mlflow.log_param("scale_pos_weight", scale_weight)

        # Hiperparámetros
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 50,
            "scale_pos_weight": scale_weight
        }
        mlflow.log_params(params)

        print("🧠 [3/5] Entrenando Modelo...")
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train)

        print("🔍 [4/5] Optimizando Umbral (Threshold Tuning)...")
        probs = clf.predict_proba(X_test)[:, 1]
        
        best_threshold = 0.5
        best_f1 = 0.0
        
        thresholds = np.arange(0.1, 0.9, 0.01)
        for thresh in thresholds:
            y_pred_thresh = (probs >= thresh).astype(int)
            f1_thresh = f1_score(y_test, y_pred_thresh)
            if f1_thresh > best_f1:
                best_f1 = f1_thresh
                best_threshold = thresh
        
        print(f"   -> MEJOR UMBRAL: {best_threshold:.2f}")
        print(f"   -> MEJOR F1: {best_f1:.4f}")

        # Generar métricas finales
        final_preds = (probs >= best_threshold).astype(int)
        acc = accuracy_score(y_test, final_preds)
        auc = roc_auc_score(y_test, probs)
        
        # Loggear todo
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("f1_score", best_f1)
        mlflow.log_param("best_threshold", best_threshold)

        print("💾 [5/5] Guardando Artefactos...")
        mlflow.xgboost.log_model(clf, "model", model_format="json")
        
        # --- AQUÍ ESTABA EL CAMBIO NECESARIO ---
        # Capturamos el ID dentro del contexto seguro
        run_id = run.info.run_id
        print(f"✅ Modelo entrenado. Run ID: {run_id}")
        
        # Guardamos el ID en el archivo para que GitHub Actions lo lea
        with open("latest_run_id.txt", "w") as f:
            f.write(run_id)
        
        print("✅ Pipeline finalizado con éxito.")

if __name__ == "__main__":
    main()