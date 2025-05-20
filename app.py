import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import shap
import matplotlib.pyplot as plt

# -- Utilitário para converter camelCase em snake_case --
def camel_to_snake(name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

# -- Configurações de página --
st.set_page_config(page_title="Detecção de Anomalias em Logs", layout="wide")
st.title("📊 Detecção e Exploração de Anomalias em Logs Financeiros")

# -- Carrega dados via API com cache --
@st.cache_data(ttl=300)
def load_logs(url: str) -> pd.DataFrame:
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    df = pd.json_normalize(data)
    df.columns = [camel_to_snake(col) for col in df.columns]
    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'])
    return df

# URL da API (configurável na sidebar)
api_url = st.sidebar.text_input(
    label="URL da API de Logs:",
    value="https://banco-facul.onrender.com/logs"
)
logs = load_logs(api_url)

# -- Exibição inicial --
st.markdown(f"**Total de registros:** {len(logs)}")
st.dataframe(logs.head(10))

# -- Pré-processamento e seleção de features --
st.sidebar.header("Pré-processamento")
all_features = [c for c in logs.columns if c not in ['id', 'descricao', 'dados_antigos', 'dados_novos', 'data']]
features = st.sidebar.multiselect(
    label="Features:",
    options=all_features,
    default=all_features
)

@st.cache_data
def preprocess(df: pd.DataFrame, features: list) -> pd.DataFrame:
    df_feat = df[features].copy()
    df_feat = pd.get_dummies(df_feat, drop_first=True)
    imputer = SimpleImputer(strategy='mean')
    df_feat[:] = imputer.fit_transform(df_feat)
    scaler = StandardScaler()
    df_feat[:] = scaler.fit_transform(df_feat)
    return df_feat

X = preprocess(logs, features)

# -- Configuração de modelos --
st.sidebar.header("Modelos de Detecção de Anomalias")
model_choice = st.sidebar.selectbox(
    label="Modelo:",
    options=["Isolation Forest"]
)
n_estimators = st.sidebar.slider(
    label="# Estimators (IF):", min_value=50, max_value=500, value=100
)
contamination = st.sidebar.slider(
    label="Contamination (%):", min_value=0.01, max_value=0.5, value=0.05
)
lof_neighbors = st.sidebar.slider(
    label="Neighbors (LOF):", min_value=5, max_value=50, value=20
)
random_state = st.sidebar.number_input(
    label="Random State:", value=42
)

# -- Treinamento e pontuação --
if model_choice == "Isolation Forest":
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state
    )
    preds = model.fit_predict(X)
    scores = model.decision_function(X)
else:
    model = LocalOutlierFactor(
        n_neighbors=lof_neighbors,
        contamination=contamination,
        novelty=True
    )
    model.fit(X)
    preds = model.predict(X)
    scores = model.decision_function(X)

logs['anomaly_score'] = scores
logs['anomaly'] = preds

# -- Explicabilidade com SHAP para IF --
st.subheader("Explicação de Anomalias (SHAP)")
if model_choice == "Isolation Forest":
    idx = st.number_input(
        label="Índice do registro para explicar:",
        min_value=0,
        max_value=len(X)-1,
        value=0
    )
    try:
        explainer = shap.TreeExplainer(model, data=X, model_output='raw')
        shap_values = explainer.shap_values(X)
        # Desenha waterfall em figura explícita
        fig_shap = plt.figure()
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value, shap_values[idx], feature_names=X.columns
        )
        st.pyplot(fig=fig_shap)
    except Exception:
        # Fallback: KernelExplainer
        background = shap.sample(X, 100)
        explainer = shap.KernelExplainer(model.decision_function, background)
        shap_values = explainer.shap_values(X.iloc[idx:idx+1])
        fig_force = shap.force_plot(
            explainer.expected_value, shap_values[0], X.iloc[idx], matplotlib=True
        )
        st.pyplot(fig=fig_force)
else:
    st.write("Explanação SHAP disponível apenas para Isolation Forest.")

# -- Visualizações de anomalias --
st.subheader("Distribuição de Scores")
fig1, ax1 = plt.subplots()
ax1.hist(logs['anomaly_score'], bins=50)
st.pyplot(fig=fig1)

st.subheader("PCA 2D para Anomalias")
# PCA agora recebe X sem NaNs
pca = PCA(n_components=2, random_state=random_state)
proj = pca.fit_transform(X)
proj_df = pd.DataFrame(proj, columns=['PC1', 'PC2'])
proj_df['anomaly'] = preds
fig2, ax2 = plt.subplots()
colors = {1: 'blue', -1: 'red'}
ax2.scatter(proj_df['PC1'], proj_df['PC2'], c=proj_df['anomaly'].map(colors), alpha=0.5)
st.pyplot(fig=fig2)

if 'data' in logs.columns:
    st.subheader("Série Temporal de Anomaly Score")
    ts = logs.set_index('data')['anomaly_score'].resample('D').mean()
    st.line_chart(ts)

st.subheader("Registros Anômalos Detectados")
cols_show = ['id', 'data'] + features + ['anomaly_score']
st.dataframe(
    logs[logs['anomaly'] == -1][cols_show]
        .sort_values('anomaly_score')
)

st.success("Análise de anomalias completa!")
