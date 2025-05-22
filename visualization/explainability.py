import matplotlib.pyplot as plt
import shap

def explain_isolation_forest(model, X, idx=0):
    """Gera explicações SHAP para um modelo Isolation Forest."""
    try:
        explainer = shap.TreeExplainer(model, data=X, model_output='raw')
        shap_values = explainer.shap_values(X)
        fig_shap = plt.figure()
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value, shap_values[idx], feature_names=X.columns
        )
        return fig_shap
    except Exception:
        # Fallback: KernelExplainer
        background = shap.sample(X, 100)
        explainer = shap.KernelExplainer(model.decision_function, background)
        shap_values = explainer.shap_values(X.iloc[idx:idx+1])
        fig_force = shap.force_plot(
            explainer.expected_value, shap_values[0], X.iloc[idx], matplotlib=True
        )
        return fig_force