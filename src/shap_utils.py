
import shap
import numpy as np

def compute_shap_values(model_predict_fn, X):
    """
    Compute SHAP values for a given model and input data.

    Parameters:
    - model_predict_fn: Callable, e.g., model.predict
    - X: pandas DataFrame (unscaled input features)

    Returns:
    - shap_values: SHAP values object
    """
    shap.initjs()
    explainer = shap.Explainer(model_predict_fn, X)
    shap_values = explainer(X)
    return shap_values

def get_top_shap_features(shap_values, feature_names, top_n=3):
    """
    Get top N most important features from SHAP values.

    Parameters:
    - shap_values: SHAP values object
    - feature_names: list of feature names (column names)
    - top_n: int, number of top features to return

    Returns:
    - List of top N feature names sorted by importance
    """
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(shap_importance)[::-1]
    top_features = [feature_names[i] for i in top_indices[:top_n]]
    return top_features
