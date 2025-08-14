"""
Task 1 (DM): Data Preprocessing and Exploration for Iris

Usage:
  python scripts/preprocessing_iris.py

Outputs images and summary stats under data_mining_notebook/artifacts
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
from utils import dm

ART = Path('data_mining_notebook') / 'artifacts'
ART.mkdir(parents=True, exist_ok=True)


def main():
    # Load/generate
    df = dm.load_or_generate('iris')

    # Preprocess
    X, y, feature_cols = dm.split_features(df)
    X_scaled = dm.scale_features(X)

    # EDA outputs
    dm.save_summary_stats(X_scaled, ART / 'summary_stats.csv')
    dm.plot_pairplot(X_scaled.join(y), hue='class', save_path=ART / 'pairplot_scaled.png')
    dm.plot_correlation_heatmap(X_scaled, save_path=ART / 'correlation_heatmap.png')
    dm.plot_boxplots_and_outliers(X_scaled, save_path=ART / 'boxplots_scaled.png')

    # Split function demo (80/20)
    X_train, X_test, y_train, y_test = dm.train_test_split_custom(X_scaled, y, test_size=0.2, shuffle=True, random_state=42)
    print('Train/Test sizes:', len(X_train), len(X_test))


if __name__ == '__main__':
    main()
