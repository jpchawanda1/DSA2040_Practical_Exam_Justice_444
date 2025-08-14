"""
Task 2 (DM): Clustering on Iris

Usage:
  python scripts/clustering_iris.py

Outputs plots/metrics under data_mining/artifacts
"""
from pathlib import Path
import sys

def _ensure_root_on_path():
    cwd = Path.cwd().resolve()
    p = cwd
    for _ in range(10):
        if (p / 'utils' / 'dm.py').exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return p
        if p.parent == p:
            break
        p = p.parent
    return cwd

_ensure_root_on_path()
from utils import dm  # local helper functions for DM tasks
import pandas as pd

ART = Path('data_mining') / 'artifacts'
ART.mkdir(parents=True, exist_ok=True)


def main():
    df = dm.load_or_generate('iris')
    X, y_true, feature_cols = dm.split_features(df)
    X_scaled = dm.scale_features(X)

    # k=3 baseline clustering and metrics
    labels3, centers3, inertia3 = dm.kmeans_fit_predict(X_scaled, 3, random_state=42)
    metrics_k3 = dm.clustering_metrics(y_true, labels3, X_scaled)
    print(f"k=3 ARI: {metrics_k3['ARI']:.4f} | Silhouette: {metrics_k3['Silhouette']:.4f}")
    pd.DataFrame(centers3).to_csv(ART / 'k3_centers.csv', index=False)

    # Experiments and elbow: evaluate multiple k values
    results = []
    for k in [2,3,4]:
        labels, centers, inertia = dm.kmeans_fit_predict(X_scaled, k, random_state=42)
        m = dm.clustering_metrics(y_true, labels, X_scaled)
        results.append({'k': k, 'ARI': m['ARI'], 'Silhouette': m['Silhouette'], 'Inertia': inertia})
    pd.DataFrame(results).to_csv(ART / 'task2_clustering_metrics.csv', index=False)

    dm.plot_elbow(X_scaled, list(range(2,9)), save_path=ART / 'elbow_curve.png', random_state=42)
    dm.plot_cluster_scatter(X_scaled, labels3, centers3,
                            x_col='petal length (cm)', y_col='petal width (cm)',
                            save_path=ART / 'clusters_scatter_petal.png')
    dm.plot_pca_clusters(X_scaled, labels3, save_path=ART / 'clusters_pca.png', random_state=42)


if __name__ == '__main__':
    main()
