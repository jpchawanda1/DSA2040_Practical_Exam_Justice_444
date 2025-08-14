"""
Task 3 (DM): Classification and Association Rules

Usage:
  python scripts/mining_iris_basket.py

Outputs: decision_tree_plot.png, top5_rules_partB.csv under data_mining_notebook/artifacts
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
import pandas as pd

ART = Path('data_mining_notebook') / 'artifacts'
ART.mkdir(parents=True, exist_ok=True)


def part_a_classification():
    df = dm.load_or_generate('iris')
    X, y, feature_cols = dm.split_features(df)
    X_scaled = dm.scale_features(X)
    X_train, X_test, y_train, y_test = dm.train_test_split_custom(X_scaled, y, test_size=0.2, shuffle=True, random_state=42)

    metrics_dt = dm.decision_tree_train_plot(
        X_train, y_train, X_test, y_test,
        feature_names=feature_cols,
        class_names=sorted(y.unique()),
        plot_path=ART / 'decision_tree_plot.png',
        max_depth=4,
    )
    print('Decision Tree metrics:', metrics_dt)

    metrics_knn = dm.knn_train_eval(X_train, y_train, X_test, y_test, k=5)
    print('KNN (k=5) metrics:', metrics_knn)

    comp = pd.DataFrame([
        {'model': 'DecisionTree', **metrics_dt},
        {'model': 'KNN(k=5)', **metrics_knn}
    ]).set_index('model')
    comp.to_csv(ART / 'classification_comparison.csv')


def part_b_association_rules():
    transactions = dm.generate_synthetic_transactions(n_transactions=40, rng_seed=42)
    rules = dm.apriori_rules(transactions, min_support=0.2, min_confidence=0.5)
    if rules is None or (hasattr(rules, 'empty') and rules.empty):
        print('No rules found at given thresholds.')
        return
    rules_sorted = rules.sort_values('lift', ascending=False).head(5).reset_index(drop=True)

    def set_to_str(x):
        if isinstance(x, (set, frozenset)):
            return ','.join(sorted(list(x)))
        return x

    out_df = rules_sorted.copy()
    if 'antecedents' in out_df.columns:
        out_df['antecedents'] = out_df['antecedents'].apply(set_to_str)
    if 'consequents' in out_df.columns:
        out_df['consequents'] = out_df['consequents'].apply(set_to_str)
    out_df.to_csv(ART / 'top5_rules_partB.csv', index=False)
    print('Saved top 5 rules to', ART / 'top5_rules_partB.csv')


def main():
    part_a_classification()
    part_b_association_rules()


if __name__ == '__main__':
    main()
