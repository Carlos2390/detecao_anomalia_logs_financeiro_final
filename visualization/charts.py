import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_score_distribution(scores):
    """Gera um histograma da distribuição de scores de anomalia."""
    fig, ax = plt.subplots()
    ax.hist(scores, bins=50)
    return fig


def plot_pca_projection(X, preds, random_state=42):
    """Gera um gráfico de projeção PCA 2D com anomalias destacadas."""
    pca = PCA(n_components=2, random_state=random_state)
    proj = pca.fit_transform(X)
    proj_df = pd.DataFrame(proj, columns=['PC1', 'PC2'])
    proj_df['anomaly'] = preds

    fig, ax = plt.subplots()
    colors = {1: 'blue', -1: 'red'}
    ax.scatter(proj_df['PC1'], proj_df['PC2'], c=proj_df['anomaly'].map(colors), alpha=0.5)
    return fig


def plot_hourly_anomalies(logs_df):
    """Visualiza a distribuição de anomalias por hora do dia."""
    # Assegurar que temos as colunas necessárias
    if 'anomaly' not in logs_df.columns or 'hora' not in logs_df.columns:
        return plt.figure()

    # Agrupar por hora e contar normais vs anômalos
    hour_anomaly = logs_df.groupby(['hora', 'anomaly']).size().unstack(fill_value=0)

    # Renomear colunas para melhor entendimento
    if -1 in hour_anomaly.columns and 1 in hour_anomaly.columns:
        hour_anomaly = hour_anomaly.rename(columns={-1: 'Anomalias', 1: 'Normais'})

    # Calcular porcentagem de anomalias
    total = hour_anomaly.sum(axis=1)
    hour_anomaly['% Anomalias'] = hour_anomaly['Anomalias'] / total * 100

    # Criar gráfico
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Barras para contagens absolutas
    hour_anomaly[['Normais', 'Anomalias']].plot(kind='bar', stacked=True,
                                                ax=ax1, color=['blue', 'red'],
                                                alpha=0.7)

    # Linha para porcentagem
    ax2 = ax1.twinx()
    hour_anomaly['% Anomalias'].plot(kind='line', marker='o',
                                     color='darkred', ax=ax2)

    # Formatação
    ax1.set_title('Distribuição de Anomalias por Hora do Dia', fontsize=14)
    ax1.set_xlabel('Hora do Dia', fontsize=12)
    ax1.set_ylabel('Quantidade de Logs', fontsize=12)
    ax2.set_ylabel('% de Anomalias', fontsize=12)

    # Adicionar linha destacando horários incomuns
    plt.axvspan(-0.5, 6.5, color='lightgrey', alpha=0.3)  # Madrugada
    plt.axvspan(18.5, 23.5, color='lightgrey', alpha=0.3)  # Noite

    plt.tight_layout()
    return fig