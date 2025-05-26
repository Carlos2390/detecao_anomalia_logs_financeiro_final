import matplotlib.pyplot as plt
import pandas as pd


def plot_login_patterns(logs_df):
    """Gera gráfico para análise de padrões de login por usuário."""
    # Filtrar apenas logs de login
    login_logs = logs_df[logs_df['tabela'] == 'usuario']
    login_logs['resultado'] = login_logs['descricao'].apply(
        lambda x: 'Sucesso' if 'SUCESSO' in x else 'Erro')

    # Verificar se a coluna user_id existe (pode ser userId nos dados originais)
    if 'user_id' not in login_logs.columns and 'userId' in login_logs.columns:
        login_logs['user_id'] = login_logs['userId']

    # Agrupar por usuário e resultado
    user_login_counts = login_logs.groupby(['user_id', 'resultado']).size().unstack(fill_value=0)

    # Verificar quais colunas existem no resultado
    plot_columns = []
    if 'Sucesso' in user_login_counts.columns:
        plot_columns.append('Sucesso')
    if 'Erro' in user_login_counts.columns:
        plot_columns.append('Erro')

    # Calcular taxa de erro
    if 'Erro' in user_login_counts.columns:
        user_login_counts['taxa_erro'] = user_login_counts['Erro'] / (
                user_login_counts['Erro'] + user_login_counts['Sucesso'])
    else:
        user_login_counts['taxa_erro'] = 0

    # Criar gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'Sucesso': 'green', 'Erro': 'red'}
    colors_to_use = [colors[col] for col in plot_columns]

    bars = user_login_counts[plot_columns].plot(
        kind='bar', ax=ax, color=colors_to_use, alpha=0.7)

    # Destacar usuários com alta taxa de erro (potenciais anomalias)
    threshold = 0.5
    for i, taxa in enumerate(user_login_counts.get('taxa_erro', [])):
        if taxa > threshold:
            ax.get_xticklabels()[i].set_color('red')
            ax.get_xticklabels()[i].set_fontweight('bold')

    ax.set_title('Padrão de Tentativas de Login por Usuário', fontsize=14)
    ax.set_ylabel('Número de Tentativas', fontsize=12)
    ax.set_xlabel('ID do Usuário', fontsize=12)
    plt.xticks(rotation=45)

    # Legenda adaptativa
    legend_labels = ['Sucessos' if col == 'Sucesso' else 'Erros' for col in plot_columns]
    plt.legend(legend_labels)

    return fig


def plot_transfer_anomalies(logs_df, anomaly_scores):
    """Visualiza transferências com anomalias destacadas por valor e hora."""
    # Filtrar transferências
    transfer_logs = logs_df[logs_df['tabela'] == 'transferencia'].copy()

    # Extrair valores das transferências
    def extract_value(data_str):
        if pd.isna(data_str):
            return None
        try:
            valor_str = data_str.split('|')[0] if '|' in data_str else ''
            return float(valor_str.split('=')[1]) if '=' in valor_str else None
        except:
            return None

    transfer_logs['valor'] = transfer_logs['dados_novos'].apply(extract_value)
    transfer_logs = transfer_logs.dropna(subset=['valor'])

    # Adicionar score de anomalia
    transfer_logs = transfer_logs.copy()
    transfer_logs['anomaly_score'] = transfer_logs.index.map(
        lambda idx: anomaly_scores[idx] if idx < len(anomaly_scores) else 0)

    # Converter data para hora do dia
    transfer_logs['hora'] = pd.to_datetime(transfer_logs['data']).dt.hour

    # Plotar gráfico de dispersão
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        transfer_logs['hora'],
        transfer_logs['valor'],
        c=transfer_logs['anomaly_score'],
        cmap='coolwarm',
        alpha=0.7,
        s=100
    )

    # Adicionar rótulos para os pontos mais anômalos
    threshold = transfer_logs['anomaly_score'].quantile(0.15)  # 15% mais anômalos
    for _, row in transfer_logs[transfer_logs['anomaly_score'] < threshold].iterrows():
        ax.annotate(
            f"ID: {row['id']}",
            (row['hora'], row['valor']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )

    # Formatação
    ax.set_title('Análise de Transferências: Valor × Hora do Dia', fontsize=14)
    ax.set_xlabel('Hora do Dia', fontsize=12)
    ax.set_ylabel('Valor da Transferência (R$)', fontsize=12)
    ax.set_xticks(range(0, 24, 2))

    # Adicionar colorbar para score de anomalia
    cbar = plt.colorbar(scatter)
    cbar.set_label('Score de Anomalia (valores menores = mais anômalo)', fontsize=10)

    # Adicionar linhas de grade
    ax.grid(True, linestyle='--', alpha=0.7)

    return fig


def plot_anomaly_summary(logs_df, preds, scores):
    """Mostra um resumo das anomalias detectadas por tipo de operação."""
    # Adicionar predições e scores ao dataframe
    df_with_anomalies = logs_df.copy()
    if len(preds) == len(df_with_anomalies):
        df_with_anomalies['anomalia'] = preds
        df_with_anomalies['score'] = scores
    else:
        return plt.figure()  # Retorna figura vazia se tamanhos não corresponderem

    # Calcular porcentagem de anomalias por tipo de operação
    anomaly_by_operation = df_with_anomalies.groupby('tipo_operacao').agg(
        total=('id', 'count'),
        anomalias=('anomalia', lambda x: (x == -1).sum()),
    )

    anomaly_by_operation['porcentagem'] = (
            anomaly_by_operation['anomalias'] / anomaly_by_operation['total'] * 100
    )

    # Ordenar por porcentagem de anomalias
    anomaly_by_operation = anomaly_by_operation.sort_values('porcentagem', ascending=False)

    # Criar gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        anomaly_by_operation.index,
        anomaly_by_operation['porcentagem'],
        color='skyblue',
        edgecolor='black',
        alpha=0.7
    )

    # Adicionar valores em cima das barras
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 1,
            f'{height:.1f}%',
            ha='center',
            fontsize=9,
        )

    # Adicionar rótulos e título
    ax.set_title('Porcentagem de Anomalias por Tipo de Operação', fontsize=14)
    ax.set_xlabel('Tipo de Operação', fontsize=12)
    ax.set_ylabel('% de Registros Anômalos', fontsize=12)
    ax.set_ylim(0, min(100, anomaly_by_operation['porcentagem'].max() * 1.2))

    # Rotacionar rótulos do eixo x
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig


def plot_login_time_patterns(logs_df):
    """Gera gráfico de padrões de horário de login para detectar acesso em horários incomuns."""
    # Filtrar logs de login
    login_logs = logs_df[logs_df['tabela'] == 'usuario'].copy()

    # Converter data para datetime e extrair hora
    login_logs['data'] = pd.to_datetime(login_logs['data'])
    login_logs['hora'] = login_logs['data'].dt.hour

    # Contar logins por hora
    login_hour_counts = login_logs.groupby('hora').size()

    # Completar horas faltantes (para ter 0-23)
    full_hours = pd.Series(0, index=range(24))
    login_hour_counts = login_hour_counts.reindex(full_hours.index).fillna(0)

    # Identificar horários fora do padrão (por exemplo, fora do horário comercial)
    comercial_hours = set(range(8, 19))  # 8h às 18h
    login_hour_counts_df = pd.DataFrame({
        'contagem': login_hour_counts,
        'tipo': ['Fora do Horário Comercial' if h not in comercial_hours else 'Horário Comercial'
                 for h in login_hour_counts.index]
    })

    # Criar gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {'Horário Comercial': 'green', 'Fora do Horário Comercial': 'orange'}

    for tipo in colors:
        subset = login_hour_counts_df[login_hour_counts_df['tipo'] == tipo]
        ax.bar(subset.index, subset['contagem'], color=colors[tipo],
               label=tipo, alpha=0.7)

    ax.set_title('Distribuição de Logins por Hora do Dia', fontsize=14)
    ax.set_xlabel('Hora do Dia', fontsize=12)
    ax.set_ylabel('Número de Logins', fontsize=12)
    ax.set_xticks(range(0, 24))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    return fig