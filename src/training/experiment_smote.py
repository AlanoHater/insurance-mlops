import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.training.preprocess import load_data, clean_data, feature_engineering

def run_experiment():
    print("🧪 CARGANDO DATOS...")
    df = feature_engineering(clean_data(load_data("data/train.csv")))
    X = df.drop(columns=['Result'], errors='ignore')
    y = df['Result']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- ESCENARIO A: TU MEJOR MODELO ACTUAL (BASELINE) ---
    print("\n🥊 ROUND 1: XGBoost con Scale_Pos_Weight (Tu Campeón)")
    # Usa los parámetros ganadores de tu Optuna/GridSearch
    params_base = {
        'objective': 'binary:logistic',
        'max_depth': 5,          # Ajusta a tus mejores params
        'learning_rate': 0.1,    # Ajusta a tus mejores params
        'n_estimators': 100,     # Ajusta a tus mejores params
        'scale_pos_weight': 7.28 # El balanceo matemático
    }
    
    clf_base = xgb.XGBClassifier(**params_base)
    clf_base.fit(X_train, y_train)
    
    probs_base = clf_base.predict_proba(X_test)[:, 1]
    # Búsqueda rápida de umbral
    best_f1_base = 0
    for t in [i/100 for i in range(10,90,5)]:
        score = f1_score(y_test, (probs_base >= t).astype(int))
        if score > best_f1_base: best_f1_base = score
        
    print(f"   -> F1 Score (Baseline): {best_f1_base:.4f}")

    # --- ESCENARIO B: SMOTE (Sin scale_pos_weight) ---
    print("\n🥊 ROUND 2: XGBoost + SMOTE (Datos Sintéticos)")
    print("   Aplicando SMOTE (esto puede tardar)...")
    
    # SMOTE solo se aplica al TRAIN, nunca al TEST
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"   Datos originales: {X_train.shape} -> Con SMOTE: {X_train_res.shape}")
    
    # Quitamos scale_pos_weight porque los datos ya están balanceados 50/50 fisicamente
    params_smote = params_base.copy()
    params_smote['scale_pos_weight'] = 1 # IMPORTANTE: Volvemos a 1
    
    clf_smote = xgb.XGBClassifier(**params_smote)
    clf_smote.fit(X_train_res, y_train_res)
    
    probs_smote = clf_smote.predict_proba(X_test)[:, 1]
    
    best_f1_smote = 0
    for t in [i/100 for i in range(10,90,5)]:
        score = f1_score(y_test, (probs_smote >= t).astype(int))
        if score > best_f1_smote: best_f1_smote = score
        
    print(f"   -> F1 Score (SMOTE): {best_f1_smote:.4f}")

    # --- VEREDICTO ---
    print("\n🏆 VEREDICTO FINAL:")
    diff = best_f1_smote - best_f1_base
    if diff > 0.02: # Solo si mejora más de un 2% vale la pena la complejidad
        print(f"✅ SMOTE GANA por {diff:.4f}. Vale la pena implementarlo.")
    else:
        print(f"❌ SMOTE NO VALE LA PENA (Diferencia: {diff:.4f}). Quédate con scale_pos_weight.")

if __name__ == "__main__":
    run_experiment()