import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from itertools import product
from src.training.preprocess import load_data, clean_data, feature_engineering

# Configuración MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("insurance-cross-selling-tuning") # Nuevo experimento para no ensuciar el de prod

def optimize_threshold(clf, X, y):
    """Encuentra el mejor umbral para un modelo dado."""
    probs = clf.predict_proba(X)[:, 1]
    best_thresh = 0.5
    best_f1 = 0.0
    
    # Búsqueda rápida de umbral
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= thresh).astype(int)
        score = f1_score(y, preds)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
    return best_thresh, best_f1

def main():
    print("🚀 Iniciando Carga de Datos...")
    df = load_data("data/train.csv")
    df = clean_data(df)
    df = feature_engineering(df)
    
    TARGET = 'Result'
    X = df.drop(columns=[TARGET], errors='ignore')
    y = df[TARGET]

    # Split para validación final
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- GRILLA DE HIPERPARÁMETROS ---
    # Define aquí qué quieres probar. Cuidado: El total de runs es la multiplicación de todo.
    # Ejemplo: 2 x 2 x 2 x 2 = 16 experimentos.
    param_grid = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'n_estimators': [50, 100],
        'scale_pos_weight': [1, 7.28] # Probamos sin balanceo (1) vs con balanceo calculado (7.28)
    }

    # Generar todas las combinaciones
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    print(f"🧪 Se ejecutarán {len(combinations)} experimentos. Esto puede tardar unos minutos...")

    for i, params in enumerate(combinations):
        run_name = f"run_{i+1}_lr{params['learning_rate']}_dp{params['max_depth']}"
        
        with mlflow.start_run(run_name=run_name):
            # Agregar parámetros fijos
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'auc'
            
            # Loggear parámetros
            mlflow.log_params(params)
            
            # Entrenamiento con Cross-Validation (Más robusto)
            # Usamos 3 folds para validar que el modelo es estable
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            clf = xgb.XGBClassifier(**params)
            
            # Calculamos AUC promedio con CV (Simulación de robustez)
            cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='roc_auc')
            mean_cv_auc = cv_scores.mean()
            mlflow.log_metric("cv_auc_mean", mean_cv_auc)
            
            # Entrenamiento final en todo el train set
            clf.fit(X_train, y_train)
            
            # Evaluación en Test Set (Hold-out)
            probs = clf.predict_proba(X_test)[:, 1]
            auc_test = roc_auc_score(y_test, probs)
            
            # Optimización de Umbral
            best_thresh, best_f1 = optimize_threshold(clf, X_test, y_test)
            
            # Loggear métricas finales
            mlflow.log_metric("test_auc", auc_test)
            mlflow.log_metric("test_f1_opt", best_f1)
            mlflow.log_param("best_threshold", best_thresh)
            
            print(f"Run {i+1}/{len(combinations)} | Params: {params} | CV AUC: {mean_cv_auc:.4f} | F1: {best_f1:.4f}")
            
            # Guardar modelo solo si es decente (ahorra espacio)
            if best_f1 > 0.35:
                mlflow.xgboost.log_model(clf, "model", model_format="json")

    print("✅ Grid Search Finalizado. Revisa MLflow UI.")

if __name__ == "__main__":
    main()