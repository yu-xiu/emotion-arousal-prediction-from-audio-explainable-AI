from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

def train_and_evaluate_models(df, feature_cols, target_col='arousal', test_size=0.2, random_state=42):
    """
    Train and evaluate 4 regression models on tabular audio data:
    Linear Regression, Random Forest, XGBoost, MLP Regressor.

    Parameters:
        df (pd.DataFrame): The input DataFrame
        feature_cols (list): List of selected feature column names
        target_col (str): Target variable name (default: 'arousal')
        test_size (float): Proportion of test set
        random_state (int): Random state for reproducibility

    Returns:
        results_df (pd.DataFrame): Evaluation results of each model
    """

    X = df[feature_cols]
    y = df[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Prepare scalers
    std_scaler = StandardScaler() #standardization
    minmax_scaler = MinMaxScaler() #normalization
    
    X_train_std = std_scaler.fit_transform(X_train)
    X_test_std = std_scaler.transform(X_test)

    X_train_minmax = minmax_scaler.fit_transform(X_train)
    X_test_minmax = minmax_scaler.transform(X_test)

    # Define models
    models = {
        "Linear Regression": (LinearRegression(), X_train_std, X_test_std),
        "Random Forest": (RandomForestRegressor(random_state=random_state), X_train, X_test),
        "XGBoost": (XGBRegressor(random_state=random_state), X_train, X_test),
        "MLP Regressor": (MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, 
                                       random_state=random_state), X_train_minmax, X_test_minmax)
    }

    results = []
    trained_models = {}
    
    for name, (model, X_tr, X_te) in models.items():
      # Fit once for test split eval
      model.fit(X_tr, y_train)
      y_pred = model.predict(X_te)

      r2 = r2_score(y_test, y_pred)
      mae = mean_absolute_error(y_test, y_pred)

      trained_models[name] = model

      # Cross-validation (on full training set)
      if name == "Linear Regression":
          cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='r2')
      elif name == "MLP Regressor":
          cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='r2')
      else:
          cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='r2')

      results.append({
          "Model": name,
          "R² (hold-out)": round(r2, 4),
          "MAE (hold-out)": round(mae, 4),
          "R² (CV Mean)": round(np.mean(cv_scores), 4)
      })

    results_df = pd.DataFrame(results)
    return results_df, trained_models
