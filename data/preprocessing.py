import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

@st.cache_data
def preprocess(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Pré-processa os dados para análise de anomalias."""

    df_processed = df.copy()

    # 1. Verificar e extrair valor da transferência se for tabela 'transferencia'
    if 'tabela' in df_processed.columns and 'dados_novos' in df_processed.columns:
        def extract_transfer_value(row):
            if row['tabela'] == 'transferencia' and isinstance(row['dados_novos'], str):
                try:
                    # Extrair o valor da string no formato 'valor=<valor>|...'
                    parts = row['dados_novos'].split('|')
                    for part in parts:
                        if part.startswith('valor='):
                            return float(part.split('=')[1])
                except (ValueError, IndexError):
                    pass # Ignorar erros de conversão ou formato
            return None # Retorna None se não for transferencia ou não encontrar valor
        df_processed['valor_transferencia'] = df_processed.apply(extract_transfer_value, axis=1)

    df_feat = df_processed[features].copy()
    df_feat = pd.get_dummies(df_feat, drop_first=True)
    imputer = SimpleImputer(strategy='mean')
    df_feat[:] = imputer.fit_transform(df_feat)
    scaler = StandardScaler()
    df_feat[:] = scaler.fit_transform(df_feat)
    return df_feat