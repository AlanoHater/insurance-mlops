import optuna
import pandas as pd
import xgboost as xgb
import mlflow
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from src.training.preprocess import load_data, clean_data, feature_engineering

# Configuración MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("insurance-optuna-bayes")

def optimize_threshold_fast(probs, y_true):
    """Búsqueda vectorial del mejor umbral (más rápida)."""
    best_f1 = 0
    best_thresh = 0.5
    # Probamos 100 umbrales en vectores numpy
    thresholds = np.linspace(0.01, 0.99, 100)
    for t in thresholds:
        preds = (probs >= t).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = score
            best_thresh = t
    return best_thresh, best_f1

def objective(trial):
    """Función objetivo que Optuna intentará maximizar."""
    
    # 1. Definir el espacio de búsqueda (Search Space)
    # Optuna sugiere valores basados en intentos anteriores
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        # Sugerir enteros o flotantes en rangos lógicos
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        # Fijo porque sabemos que funciona bien
        'scale_pos_weight': 7.28 
    }

    # 2. Carga de Datos (Cacheada en memoria si fuera necesario, aquí simple)
    # Nota: En producción real cargaríamos datos FUERA de la función objective
    # para no leer disco 50 veces. Lo dejo aquí por simplicidad del script.
    df = load_data("data/train.csv")
    df = clean_data(df)
    df = feature_engineering(df)
    TARGET = 'Result'
    X = df.drop(columns=[TARGET], errors='ignore')
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Entrenar Modelo con Pruning (Poda)
    # Optuna puede detener entrenamientos malos a la mitad para ahorrar tiempo
    clf = xgb.XGBClassifier(**param)
    
    # Iniciar Run de MLflow anidado
    with mlflow.start_run(nested=True):
        clf.fit(X_train, y_train)
        
        # Predecir y Optimizar Umbral
        probs = clf.predict_proba(X_test)[:, 1]
        best_thresh, best_f1 = optimize_threshold_fast(probs, y_test)
        
        # Loggear a MLflow
        mlflow.log_params(param)
        mlflow.log_metric("f1_opt", best_f1)
        mlflow.log_metric("threshold", best_thresh)
        
        # Importante: Guardar el modelo si es el mejor hasta ahora
        trial.set_user_attr("best_model", clf)
        
    return best_f1

def main():
    print("🧠 Iniciando Optimización Bayesiana con Optuna...")
    
    # Crear estudio: Queremos MAXIMIZAR el F1 Score
    study = optuna.create_study(direction="maximize")
    
    # Ejecutar 20 intentos (trials)
    # Optuna es inteligente: 20 trials valen por 100 de Grid Search
    study.optimize(objective, n_trials=20)

    print("-" * 50)
    print("🏆 MEJORES RESULTADOS:")
    print(f"Mejor F1: {study.best_value:.4f}")
    print(f"Mejores Params: {study.best_params}")
    print("-" * 50)
    
    # Guardar el mejor modelo en MLflow explícitamente al final
    print("Guardando el modelo campeón...")
    with mlflow.start_run(run_name="Best_Bayesian_Model"):
        best_params = study.best_params
        best_params['scale_pos_weight'] = 7.28 # Asegurar que esté incluido
        mlflow.log_params(best_params)
        mlflow.log_metric("f1_score", study.best_value)
        
        # Re-entrenar o recuperar el modelo para guardarlo
        # (Por simplicidad, aquí re-entrenamos rápido con los mejores params)
        df = load_data("data/train.csv")
        df = feature_engineering(clean_data(df))
        X = df.drop(columns=['Result'], errors='ignore')
        y = df['Result']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        clf = xgb.XGBClassifier(**best_params)
        clf.fit(X_train, y_train)
        
        mlflow.xgboost.log_model(clf, "model", model_format="json")
        print("✅ Modelo Campeón Registrado.")

if __name__ == "__main__":
    main()