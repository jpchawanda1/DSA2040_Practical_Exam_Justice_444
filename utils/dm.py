from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier


# ------------------------
# Data loading / generation
# ------------------------

def load_iris_dataframe() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.rename(columns={'target': 'class'}, inplace=True)
    mapping = {i: name for i, name in enumerate(iris.target_names)}
    df['class'] = df['class'].map(mapping)
    return df


def generate_synthetic_dataframe(n_per_class: int = 50) -> pd.DataFrame:
    specs = [
        ('setosa_like',   [5.0, 3.5, 1.5, 0.25], [0.25, 0.20, 0.15, 0.05]),
        ('versicolor_like',[6.0, 2.8, 4.2, 1.30], [0.30, 0.25, 0.30, 0.10]),
        ('virginica_like', [6.5, 3.0, 5.5, 2.00], [0.35, 0.25, 0.35, 0.15])
    ]
    rows = []
    for label, means, stds in specs:
        data = np.column_stack([np.random.normal(m, s, n_per_class) for m, s in zip(means, stds)])
        for r in data:
            rows.append({
                'sepal length (cm)': r[0],
                'sepal width (cm)':  r[1],
                'petal length (cm)': r[2],
                'petal width (cm)':  r[3],
                'class': label,
            })
    return pd.DataFrame(rows)


def load_or_generate(data_option: str = 'iris') -> pd.DataFrame:
    return load_iris_dataframe() if data_option == 'iris' else generate_synthetic_dataframe()


# ------------------------
# Preprocessing & EDA helpers
# ------------------------

def split_features(df: pd.DataFrame, target_col: str = 'class') -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    feature_cols = [c for c in df.columns if c != target_col]
    return df[feature_cols].copy(), df[target_col].copy(), feature_cols


def scale_features(X: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled


def one_hot_encode(series: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
    try:
        ohe = OneHotEncoder(sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(sparse=False)
    arr = ohe.fit_transform(series.to_frame())
    if hasattr(arr, 'toarray'):
        arr = arr.toarray()
    encoded = pd.DataFrame(arr, columns=[f'class_{c}' for c in ohe.categories_[0]])
    return encoded, list(ohe.categories_[0])


def save_summary_stats(X: pd.DataFrame, path: Path) -> pd.DataFrame:
    summary = X.describe().T
    summary.to_csv(path, index=True)
    return summary


def plot_pairplot(df: pd.DataFrame, hue: str, save_path: Path) -> None:
    sns.set_theme(style='whitegrid', context='notebook')
    pp = sns.pairplot(df, hue=hue, corner=True, diag_kind='hist')
    pp.fig.suptitle('Pairplot of Scaled Features by Class', y=1.02)
    pp.savefig(save_path, dpi=150, bbox_inches='tight')


def plot_correlation_heatmap(X: pd.DataFrame, save_path: Path) -> None:
    corr = X.corr()
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={'shrink':0.8})
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)


def plot_boxplots_and_outliers(X_scaled: pd.DataFrame, save_path: Path) -> Dict[str, int]:
    feature_cols = list(X_scaled.columns)
    fig, axes = plt.subplots(1, len(feature_cols), figsize=(4*len(feature_cols), 4))
    if len(feature_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, feature_cols):
        sns.boxplot(y=X_scaled[col], ax=ax, color='#4c72b0')
        ax.set_title(col)
    plt.suptitle('Boxplots (Scaled Features)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)

    outlier_summary: Dict[str, int] = {}
    for col in feature_cols:
        q1, q3 = X_scaled[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        outlier_summary[col] = int(((X_scaled[col] < lower) | (X_scaled[col] > upper)).sum())
    return outlier_summary


def train_test_split_custom(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, shuffle: bool = True, random_state: Optional[int] = None):
    assert 0 < test_size < 1, 'test_size must be between 0 and 1'
    if shuffle:
        rng = np.random.default_rng(random_state)
        indices = np.arange(len(X))
        rng.shuffle(indices)
        X_shuff = X.iloc[indices].reset_index(drop=True)
        y_shuff = y.iloc[indices].reset_index(drop=True)
    else:
        X_shuff, y_shuff = X.reset_index(drop=True), y.reset_index(drop=True)
    split_idx = int(len(X) * (1 - test_size))
    return X_shuff.iloc[:split_idx], X_shuff.iloc[split_idx:], y_shuff.iloc[:split_idx], y_shuff.iloc[split_idx:]


# ------------------------
# Clustering helpers
# ------------------------

def kmeans_fit_predict(X_scaled: pd.DataFrame, k: int, random_state: int = 42) -> Tuple[np.ndarray, pd.DataFrame, float]:
    km = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
    labels = km.fit_predict(X_scaled)
    centers = pd.DataFrame(km.cluster_centers_, columns=X_scaled.columns)
    inertia = float(km.inertia_)
    return labels, centers, inertia


def clustering_metrics(y_true: pd.Series, labels: np.ndarray, X_scaled: pd.DataFrame) -> Dict[str, float]:
    ari = float(adjusted_rand_score(y_true, labels))
    sil = float(silhouette_score(X_scaled, labels))
    return {'ARI': ari, 'Silhouette': sil}


def plot_elbow(X_scaled: pd.DataFrame, ks: List[int], save_path: Path, random_state: int = 42) -> List[float]:
    inertias: List[float] = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))
    plt.figure(figsize=(5,4))
    plt.plot(list(ks), inertias, marker='o')
    plt.title('Elbow Curve (Inertia vs k)')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.xticks(list(ks))
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    return inertias


def plot_cluster_scatter(X_scaled: pd.DataFrame, labels: np.ndarray, centers: pd.DataFrame, x_col: str, y_col: str, save_path: Path) -> None:
    plt.figure(figsize=(6,5))
    plt.scatter(X_scaled[x_col], X_scaled[y_col], c=labels, cmap='viridis', alpha=0.8, edgecolor='k')
    plt.scatter(centers[x_col], centers[y_col], c='red', s=120, marker='X', label='Centers')
    plt.xlabel(f'{x_col} (scaled)')
    plt.ylabel(f'{y_col} (scaled)')
    plt.title('K-Means Clusters')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)


def plot_pca_clusters(X_scaled: pd.DataFrame, labels: np.ndarray, save_path: Path, random_state: int = 42) -> float:
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    explained = float(pca.explained_variance_ratio_.sum())
    plt.figure(figsize=(6,5))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis', edgecolor='k', alpha=0.85)
    plt.title(f'PCA Projection VarExpl: {explained:.2%}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    return explained


# ------------------------
# Classification helpers
# ------------------------

def decision_tree_train_plot(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, feature_names: List[str], class_names: List[str], plot_path: Path, max_depth: int = 4) -> Dict[str, float]:
    dt = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = float(accuracy_score(y_test, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    plt.figure(figsize=(10,6))
    plot_tree(dt, feature_names=feature_names, class_names=sorted(class_names), filled=True, rounded=True)
    plt.title(f'Decision Tree (max_depth={max_depth})')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    return {'accuracy': acc, 'precision': float(prec), 'recall': float(rec), 'f1': float(f1)}


def knn_train_eval(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, k: int = 5) -> Dict[str, float]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = float(accuracy_score(y_test, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    return {'accuracy': acc, 'precision': float(prec), 'recall': float(rec), 'f1': float(f1)}


# ------------------------
# Association rules helpers
# ------------------------

def generate_synthetic_transactions(n_transactions: int = 40, rng_seed: int = 42) -> List[List[str]]:
    item_pool = ['milk','bread','butter','cheese','eggs','beer','diapers','apples','bananas','cereal',
                 'chicken','rice','pasta','tomatoes','onions','yogurt','chips','soda','coffee','tea']
    rng = np.random.default_rng(rng_seed)
    transactions: List[List[str]] = []
    for _ in range(n_transactions):
        basket_size = rng.integers(3,9)
        basket = rng.choice(item_pool, size=basket_size, replace=False).tolist()
        if rng.random() < 0.4:
            for it in ['milk','bread']:
                if it not in basket: basket.append(it)
        if rng.random() < 0.25:
            for it in ['beer','diapers']:
                if it not in basket: basket.append(it)
        if rng.random() < 0.3:
            for it in ['coffee','tea']:
                if it not in basket: basket.append(it)
        transactions.append(sorted(set(basket)))
    return transactions


def apriori_rules(transactions: List[List[str]], min_support: float = 0.2, min_confidence: float = 0.5) -> pd.DataFrame:
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_tx = pd.DataFrame(te_ary, columns=te.columns_)
        frequent = apriori(df_tx, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent, metric='confidence', min_threshold=min_confidence)
        # Standardize core columns
        cols = ['antecedents','consequents','support','confidence','lift']
        for c in cols:
            if c not in rules.columns:
                rules[c] = np.nan
        return rules[cols]
    except Exception:
        # Fallback: simple pairs-only apriori
        def support(itemset: set[str]) -> float:
            count = sum(1 for t in transactions if set(itemset).issubset(t))
            return count / len(transactions)
        singles = sorted({i for t in transactions for i in t})
        L1 = [{i} for i in singles if support({i}) >= min_support]
        pairs: List[Tuple[set[str], float]] = []
        for i in range(len(L1)):
            for j in range(i+1, len(L1)):
                cand = L1[i] | L1[j]
                sup = support(cand)
                if sup >= min_support:
                    pairs.append((cand, sup))
        rows = []
        for cand, sup in pairs:
            a,b = tuple(cand)
            sup_a, sup_b = support({a}), support({b})
            conf_a_b = sup / sup_a if sup_a else 0.0
            conf_b_a = sup / sup_b if sup_b else 0.0
            lift_a_b = conf_a_b / sup_b if sup_b else 0.0
            lift_b_a = conf_b_a / sup_a if sup_a else 0.0
            if conf_a_b >= min_confidence:
                rows.append({'antecedents': {a}, 'consequents': {b}, 'support': sup, 'confidence': conf_a_b, 'lift': lift_a_b})
            if conf_b_a >= min_confidence:
                rows.append({'antecedents': {b}, 'consequents': {a}, 'support': sup, 'confidence': conf_b_a, 'lift': lift_b_a})
        return pd.DataFrame(rows)


# ------------------------
# Utilities
# ------------------------

def save_json(obj: dict, path: Path) -> None:
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
