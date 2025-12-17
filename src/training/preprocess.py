import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def load_data(path):
    """Carga datos validando existencia."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"El archivo {path} no existe.")
    return pd.read_csv(path)

def clean_data(df):
    """Limpieza técnica: formatos, nulos y tipos."""
    # 1. Eliminar ID (Ruido)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # 2. Limpiar Moneda (AnnualPremium: '£1,200' -> 1200.0)
    if 'AnnualPremium' in df.columns and df['AnnualPremium'].dtype == 'object':
        df['AnnualPremium'] = df['AnnualPremium'].astype(str).str.replace('£', '').str.replace(',', '')
        df['AnnualPremium'] = pd.to_numeric(df['AnnualPremium'], errors='coerce')
        
    # 3. Imputación de Nulos (Estrategia MVP)
    # Numéricas -> Mediana
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Categóricas -> Moda
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    return df

def feature_engineering(df):
    """Transformación de variables para XGBoost."""
    df = df.copy()
    
    # 1. Mapping Ordinal (VehicleAge)
    age_map = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
    if 'VehicleAge' in df.columns:
        df['VehicleAge'] = df['VehicleAge'].map(age_map).fillna(-1)
        
    # 2. Encoding de Categóricas Nominales
    # XGBoost requiere números. Usamos LabelEncoder por eficiencia.
    le = LabelEncoder()
    cols_to_encode = ['PastAccident', 'Vehicle_Damage', 'Switch', 'Gender'] 
    
    for col in cols_to_encode:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    
    return df