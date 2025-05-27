import streamlit as st
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import matplotlib.pyplot as plt

from algorithms.isolation_forest import train_isolation_forest
from data.loader import load_logs
from data.preprocessing import preprocess, preprocess_transfer_values, extract_time_features
from visualization.charts import plot_score_distribution, plot_pca_projection, plot_hourly_anomalies
from visualization.explainability import explain_isolation_forest
from visualization.financial_charts import (
    plot_login_patterns,
    plot_transfer_anomalies,
    plot_anomaly_summary,
    plot_login_time_patterns
)

# -- Configurações de página --
st.set_page_config(page_title="Detecção de Anomalias em Logs", layout="wide", page_icon="./icone/icone_bank_sentinel.png")

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.write("")

with col2:
    st.image("./icone/icone_bank_sentinel.png")

with col3:
    st.write("")

st.title("Detecção e Exploração de Anomalias em Logs Financeiros")

# -- Sidebar Config --
st.sidebar.header("Configurações de Dados")
api_url = st.sidebar.text_input(
    label="URL da API de Logs:",
    value="https://banco-facul.onrender.com/logs"
)

# Configuração de recarga automática
st.sidebar.header("Recarga Automática")
auto_reload = st.sidebar.checkbox("Ativar recarga automática", value=False)
reload_interval = st.sidebar.slider(
    "Intervalo de recarga (segundos)",
    min_value=10,
    max_value=600,
    value=60,
    step=10
)

# Recarregar automaticamente usando st_autorefresh
if auto_reload:
    load_logs.clear()
    count = st_autorefresh(interval=reload_interval * 1000, key="auto_reload")
    st.sidebar.write(f"🔄 Página recarregada automaticamente {count} vezes.")
else:
    st.sidebar.write("✅ Recarga automática desativada.")

# Botão de recarregar manualmente
if st.sidebar.button("🔄 Recarregar Logs Agora"):
    load_logs.clear()
    st.rerun()

# Exibir momento da última atualização
st.sidebar.markdown("---")
st.sidebar.write(f"**Última atualização:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

# -- Carregar os logs --
logs = load_logs(api_url)
logs = preprocess_transfer_values(logs)
logs = extract_time_features(logs)

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

#st.subheader("PCA 2D para Anomalias")
#fig_pca = plot_pca_projection(X, preds, random_state)
#st.pyplot(fig=fig_pca)

if 'data' in logs.columns:
    st.subheader("Série Temporal de Anomaly Score")

    # Filtrar apenas anomalias (anomaly = -1)
    anomalies_only = logs[logs['anomaly'] == -1].copy()

    # Criar figura com Matplotlib para maior controle
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plotar cada anomalia como um ponto
    ax.scatter(anomalies_only['data'], anomalies_only['anomaly_score'],
               color='red', s=50, alpha=0.7)

    # Adicionar linha conectando os pontos para visualizar tendência
    ax.plot(anomalies_only['data'], anomalies_only['anomaly_score'],
            color='gray', alpha=0.5, linestyle='--')

    # Formatação
    ax.set_title('Série Temporal de Anomaly Score (Anomalias Individuais)')
    ax.set_ylabel('Anomaly Score')
    ax.set_xlabel('Data')

    # Rotacionar datas para melhor visualização
    plt.xticks(rotation=45)

    # Adicionar grade
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Exibir gráfico
    st.pyplot(fig)

st.subheader("Registros Anômalos Detectados")

# Use um conjunto para garantir unicidade das colunas
cols_set = {'id', 'data', 'anomaly_score'}

# Adicionar colunas de features
cols_set.update([col for col in features if col in logs.columns])

# Adicionar 'dados_novos' se a tabela for 'transferencia' (embora o filtro seja feito depois, a coluna precisa estar presente para visualização)
if 'dados_novos' in logs.columns:
    cols_set.add('dados_novos')

# Converter o conjunto de volta para uma lista para manter uma ordem razoável (essenciais primeiro)
cols_show = ['id', 'data', 'anomaly_score'] + sorted(list(cols_set - {'id', 'data', 'anomaly_score'}))

st.dataframe(
    logs[logs['anomaly'] == -1][cols_show]
    .sort_values('anomaly_score')
)

# Adicione esta seção antes do st.success()
st.subheader("🕰️ Análise de Anomalias por Horário")
fig_hour_anomalies = plot_hourly_anomalies(logs)
st.pyplot(fig=fig_hour_anomalies)

st.success("Análise de anomalias completa!")

st.header("📈 Análises Específicas para Contexto Financeiro")

st.subheader("🔐 Análise de Padrões de Login")
fig_login = plot_login_patterns(logs)
st.pyplot(fig=fig_login)

st.subheader("💸 Detecção de Transferências Anômalas")
fig_transfer = plot_transfer_anomalies(logs, scores)
st.pyplot(fig=fig_transfer)

st.subheader("📊 Resumo de Anomalias por Operação")
fig_summary = plot_anomaly_summary(logs, preds, scores)
st.pyplot(fig=fig_summary)

st.subheader("🕒 Análise de Horários de Login")
fig_login_time = plot_login_time_patterns(logs)
st.pyplot(fig=fig_login_time)
