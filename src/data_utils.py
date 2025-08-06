# src/plot_utils.py

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib


def plot_top_correlations(df, target_col, top_n=10):
    """
    Plot a bar chart of the top N features most correlated with the target column.

    Parameters:
    - df: pandas.DataFrame  
        The input DataFrame containing both features and the target variable.
    - target_col: str  
        The name of the target column in `df`.
    - top_n: int, optional (default=10)  
        The number of top features (based on absolute correlation) to display in the plot.

    Returns:
    - None. Displays a matplotlib bar plot showing the top correlated features.
    """
    # Compute absolute correlation and sort
    corr = (
        df.corr()[target_col]
        .drop(target_col)
        .abs()
        .sort_values(ascending=False)
        .head(top_n)
    )

    # Plot
    plt.figure(figsize=(12, 6))  # changed from (50, 3) to more standard size
    sns.barplot(x=corr.values, y=corr.index, palette="viridis")
    plt.title(f"Top {top_n} Features Correlated with '{target_col}'", fontsize=14)
    plt.xlabel("Absolute Correlation")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def plot_target_feature_heatmap(df, target_col, corr_threshold=0.3):
    """
    Plot the heatmap for target and features with correlation threshold of 0.3.

    Parameters:
    - df: pandas.DataFrame  
        The input DataFrame containing both features and the target variable.
    - target_col: str  
        The name of the target column in `df`.
    - corr_threshold: float, optional (default=0.3)  
        Correlation threshold with 0.3.

    Returns:
    - None. Displays the heatmap
    """
    corr_with_target = df.corr()[target_col].abs()
    selected = corr_with_target[corr_with_target > corr_threshold].index.tolist()
    selected = [col for col in selected if col != target_col]
    selected_df = df[selected]
    corr_matrix = selected_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
    plt.title(f"Correlation Heatmap of Top Features for {target_col}")
    plt.tight_layout()
    plt.show()


def select_target_related_features(df, target_col, corr_threshold=0.3):
    """
    Parameters:
    - df: pandas DataFrame
      The input DataFrame containing both features and the target variable.
    - target_col: str
      The name of the target column in `df` with which to compute correlations
    - corr_threshold: float digit, optional (default = 0.3)
      The minimum absolute correlation value required for a feature to be selected
    Returns:
    - List of column names (features) that have a correlation greater than `corr_threshold` 
      (in absolute value) with the target column
    """
    corr_with_target = df.corr()[target_col].abs().sort_values(ascending=False)
    selected = corr_with_target[corr_with_target > corr_threshold].index.tolist()
    if target_col in selected:
        selected.remove(target_col)
    return selected


def remove_redundant_features(df, selected_features, redundancy_threshold=0.8):
    """
    Remove redundant features from a list of selected features based on their pairwise correlation.

    Parameters:
    - df: pandas.DataFrame  
        The input DataFrame containing the feature columns.
    - selected_features: list of str  
        A list of feature column names to evaluate for redundancy.
    - redundancy_threshold: float, optional (default=0.8)  
        The absolute correlation threshold above which two features are considered redundant.
        If the correlation between two features exceeds this threshold, one of them will be removed.

    Returns:
    - List of feature names with redundant features removed based on the specified threshold.
    """

    sub_df = df[selected_features]
    corr_matrix = sub_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > redundancy_threshold)]
    return sub_df.drop(columns=to_drop), to_drop


def plot_feature_vs_target(df, feature_col, target_col, figsize=(8, 5), color='blue', add_reg_line=False):
    """
    Parameters:
    - df: pandas DataFrame
    - feature_col: str, feature name (x)
    - target_col: str, target (y)
    - figsize: tuple, image size
    - color: str, color of points
    - add_reg_line: bool, linear regression line
    """
    plt.figure(figsize=figsize)

    if add_reg_line:
        sns.regplot(data=df, x=feature_col, y=target_col, scatter_kws={"s": 15, "alpha": 0.6}, line_kws={"color": "red"})
    else:
        plt.scatter(df[feature_col], df[target_col], color=color, s=15, alpha=0.6)

    plt.xlabel(feature_col)
    plt.ylabel(target_col)
    plt.title(f'{feature_col} vs {target_col}')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_top_correlated_features(df, target_col, top_n=6, save_path=None):
    corr = df.corr()[target_col].drop(target_col).sort_values(key=abs, ascending=False)
    top_features = corr.head(top_n).index.tolist()
    
    # plot
    for feat in top_features:
        plot_feature_vs_target(df, feat, target_col, add_reg_line=True)
    
    # save top features
    if save_path is not None:
        joblib.dump(top_features, save_path)
        print(f"Top {top_n} features saved to {save_path}")
    
    return top_features
