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