import pandas as pd
import requests
import streamlit as st
from utils.text_utils import camel_to_snake

@st.cache_data(ttl=300)
def load_logs(url: str) -> pd.DataFrame:
    """Carrega dados de logs a partir de uma API REST."""
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    df = pd.json_normalize(data)
    df.columns = [camel_to_snake(col) for col in df.columns]
    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'])
    return df