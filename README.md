# 📊 Detecção de Anomalias em Logs Financeiros

Uma aplicação Streamlit para detecção e análise de anomalias em logs de operações financeiras utilizando modelo de Machine Learning e técnicas de interpretabilidade.

![Design sem nome (3)](https://github.com/user-attachments/assets/f6785eb4-f4f2-4a6d-bdd6-53a8670400af)

## ✨ Funcionalidades

- **Conexão com API**: Carrega dados em tempo real de endpoints REST
- **Pré-processamento Automático**:
  - Conversão de features categóricas
  - Imputação de valores faltantes
  - Normalização com StandardScaler
- **Visualizações Interativas**:
  - Histograma de scores de anomalia
  - Projeção PCA 2D para análise multivariada
  - Gráfico temporal
- **Explicabilidade**:
  - Explicações locais com SHAP

## 🚀 Como Executar

1. **Pré-requisitos**:
   - Python 3.8+
   - Gerenciador de pacotes (pip/conda)

2. **Instalação**:
```bash
    git clone https://github.com/Carlos2390/detecao_anomalia_logs_financeiro_final.git
    cd deteccao-anomalias-logs_final
    pip install -r requirements.txt
```
3. **Execução**:
```bash
    streamlit run app.py
```
4. **Configuração**:
 - Insira a URL da API na sidebar (ex: https://banco-facul.onrender.com/logs)
 - Selecione features e ajuste parâmetros


## Sistema financeiro de exemplo para geração de logs:  https://kzmp5s414cv9om89o0uo.lite.vusercontent.net
