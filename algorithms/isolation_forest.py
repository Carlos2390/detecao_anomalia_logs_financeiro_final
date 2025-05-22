from sklearn.ensemble import IsolationForest


def train_isolation_forest(X, n_estimators=100, contamination=0.05, random_state=42):
    """Treina um modelo Isolation Forest para detecção de anomalias."""
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state
    )
    preds = model.fit_predict(X)
    scores = model.decision_function(X)

    return model, preds, scores