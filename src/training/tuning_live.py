import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from itertools import product
from src.training.preprocess import load_data, clean_data, feature_engineering

# Configuración
mlflow.set_tracking_uri("sqlite:///mlflow.db")
# Usamos un nombre nuevo para ver limpio el proceso
mlflow.set_experiment("insurance-live-tuning-visual") 

def optimize_threshold(clf, X, y):
    """Calcula métricas finales con el mejor umbral."""
    probs = clf.predict_proba(X)[:, 1]
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in [i/100 for i in range(10, 90, 5)]:
        preds = (probs >= thresh).astype(int)
        score = f1_score(y, preds)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
    return best_thresh, best_f1

def main():
    # 1. Autologging ON: Esto enviará las curvas en tiempo real
    mlflow.xgboost.autolog(log_input_examples=False, log_model_signatures=False, silent=True)

    print("📊 Cargando datos...")
    df = load_data("data/train.csv")
    df = clean_data(df)
    df = feature_engineering(df)
    
    TARGET = 'Result'
    X = df.drop(columns=[TARGET], errors='ignore')
    y = df[TARGET]
    
    # Split Fijo (Train / Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Definir Grilla (Usamos tus mejores rangos basados en el run anterior)
    param_grid = {
        'learning_rate': [0.05, 0.1],      # Velocidad de aprendizaje
        'max_depth': [3, 5],               # Complejidad
        'n_estimators': [100, 200],        # Duración (más largo para ver curvas)
        'scale_pos_weight': [7.28]         # Ya sabemos que el balanceo funciona, lo dejamos fijo
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    print(f"🎬 Iniciando Show: {len(combinations)} experimentos visuales...")

    for i, params in enumerate(combinations):
        # Nombre descriptivo para encontrarlo fácil en la UI
        run_name = f"Live_LR{params['learning_rate']}_D{params['max_depth']}_E{params['n_estimators']}"
        
        with mlflow.start_run(run_name=run_name):
            print(f"▶️ Ejecutando {i+1}/{len(combinations)}: {run_name}")
            
            # Parámetros base
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = ['auc', 'logloss'] # Métricas para graficar
            
            # Entrenamiento
            clf = xgb.XGBClassifier(**params)
            
            # PASO CRÍTICO: eval_set alimenta las gráficas en vivo
            clf.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False # Silencio en terminal, ruido en MLflow UI
            )
            
            # Post-Procesamiento (Cálculo de F1 Óptimo)
            best_thresh, best_f1 = optimize_threshold(clf, X_test, y_test)
            mlflow.log_metric("test_f1_opt", best_f1)
            mlflow.log_param("best_threshold", best_thresh)
            
            print(f"   ✅ Terminado. F1 Opt: {best_f1:.4f}")

if __name__ == "__main__":
    main()