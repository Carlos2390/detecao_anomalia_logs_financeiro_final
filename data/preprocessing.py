import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

@st.cache_data
def preprocess(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Pré-processa os dados para análise de anomalias."""
    df_feat = df[features].copy()
    df_feat = pd.get_dummies(df_feat, drop_first=True)
    imputer = SimpleImputer(strategy='mean')
    df_feat[:] = imputer.fit_transform(df_feat)
    scaler = StandardScaler()
    df_feat[:] = scaler.fit_transform(df_feat)
    return df_feat