"""
Task 2: ETL Process Implementation

Usage:
  python scripts/etl_retail.py

Reads Online Retail Excel from raw_data/Online Retail.xlsx, transforms, and loads into data_warehouse/retail_dw.db
"""
from pathlib import Path
import sys
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    markers = ["utils", "raw_data", ".git", "data_warehouse"]
    p = cwd
    for _ in range(10):
        if any((p / m).exists() for m in markers):
            return p
        if p.parent == p:
            break
        p = p.parent
    return cwd


def main():
    root = find_project_root()
    # Ensure project root is on sys.path for 'utils' package
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from utils.etl import run_etl  # import after sys.path update
    data_dir = root / 'raw_data'
    excel_path = data_dir / 'Online Retail.xlsx'
    db_path = root / 'data_warehouse' / 'retail_dw.db'
    fixed_current_date = pd.Timestamp('2025-08-12')

    print('ROOT:', root)
    print('Excel:', excel_path)
    print('DB:', db_path)

    counts = run_etl(excel_path, db_path, fixed_current_date)
    print('ETL complete. Counts:', counts)


if __name__ == '__main__':
    main()
