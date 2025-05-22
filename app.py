import streamlit as st

from algorithms.isolation_forest import train_isolation_forest
from data.loader import load_logs
from data.preprocessing import preprocess
from visualization.charts import plot_score_distribution, plot_pca_projection
from visualization.explainability import explain_isolation_forest

# -- Configurações de página --
st.set_page_config(page_title="Detecção de Anomalias em Logs", layout="wide")
st.title("📊 Detecção e Exploração de Anomalias em Logs Financeiros")

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

X = preprocess(logs, features)

# -- Configuração do modelo Isolation Forest --
st.sidebar.header("Modelo de Detecção de Anomalias")
n_estimators = st.sidebar.slider(
    label="# Estimators:", min_value=50, max_value=500, value=100
)
contamination = st.sidebar.slider(
    label="Contamination (%):", min_value=0.01, max_value=0.5, value=0.05
)
random_state = st.sidebar.number_input(
    label="Random State:", value=42
)

# -- Treinamento e pontuação --
model, preds, scores = train_isolation_forest(
    X, n_estimators=n_estimators, contamination=contamination, random_state=random_state
)

logs['anomaly_score'] = scores
logs['anomaly'] = preds

# -- Explicabilidade com SHAP --
st.subheader("Explicação de Anomalias (SHAP)")
idx = st.number_input(
    label="Índice do registro para explicar:",
    min_value=0,
    max_value=len(X) - 1,
    value=0
)
fig_shap = explain_isolation_forest(model, X, idx)
st.pyplot(fig=fig_shap)

# -- Visualizações de anomalias --
st.subheader("Distribuição de Scores")
fig_hist = plot_score_distribution(logs['anomaly_score'])
st.pyplot(fig=fig_hist)

st.subheader("PCA 2D para Anomalias")
fig_pca = plot_pca_projection(X, preds, random_state)
st.pyplot(fig=fig_pca)

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
