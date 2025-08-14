"""
OLAP helper functions operating on the retail star schema in SQLite.

Includes a resilient DB connector and three example analysis queries:
roll-up by country/quarter, drill-down by country/month, and a category slice.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3
import pandas as pd


def ensure_db(db_candidates: list[Path]) -> sqlite3.Connection:
    """Return a sqlite3 connection to the first existing candidate DB path."""
    for p in db_candidates:
        if p.exists():
            return sqlite3.connect(p.resolve())
    raise FileNotFoundError("Could not locate retail_dw.db; run ETL first.")


def has_product_dim(conn: sqlite3.Connection) -> bool:
    """Check if ProductDim is present in the connected database."""
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name='ProductDim';"
    return pd.read_sql(q, conn).shape[0] == 1


def rollup_country_quarter(conn: sqlite3.Connection) -> pd.DataFrame:
    """Aggregate total sales by Country, Year, Quarter (roll-up)."""
    sql = '''
    SELECT sf.Country, t.Year, t.Quarter, ROUND(SUM(sf.TotalSales),2) AS total_sales
    FROM SalesFact sf
    JOIN TimeDim t ON sf.DateKey = t.DateKey
    GROUP BY sf.Country, t.Year, t.Quarter
    ORDER BY total_sales DESC;
    '''
    return pd.read_sql(sql, conn)


def drilldown_country_month(conn: sqlite3.Connection, country: str) -> pd.DataFrame:
    """Monthly trend for a specific country (drill-down)."""
    sql = '''
    SELECT t.Year, t.Month, SUM(sf.TotalSales) AS monthly_sales
    FROM SalesFact sf
    JOIN TimeDim t ON sf.DateKey = t.DateKey
    WHERE sf.Country = ?
    GROUP BY t.Year, t.Month
    ORDER BY t.Year, t.Month;
    '''
    return pd.read_sql(sql, conn, params=[country])


def slice_category_totals(conn: sqlite3.Connection) -> pd.DataFrame:
    """Slice total sales by product category; falls back if ProductDim is absent."""
    if has_product_dim(conn):
        sql = '''
        SELECT p.Category AS category, ROUND(SUM(f.TotalSales),2) AS total_sales
        FROM SalesFact f
        JOIN ProductDim p ON f.ProductKey = p.ProductKey
        GROUP BY p.Category
        ORDER BY total_sales DESC;
        '''
        return pd.read_sql(sql, conn)
    else:
        # Fallback: build a temporary mapping from descriptions if ProductDim is absent
        category_sql = '''
        DROP TABLE IF EXISTS ProductCategory;
        CREATE TEMP TABLE ProductCategory AS
        SELECT DISTINCT
          StockCode,
          COALESCE(Description,'') AS Description,
          CASE
            WHEN UPPER(Description) LIKE '%LED%' OR UPPER(Description) LIKE '%LIGHT%' OR
                 UPPER(Description) LIKE '%LAMP%' OR UPPER(Description) LIKE '%ELECTRIC%' OR
                 UPPER(Description) LIKE '%BATTERY%' THEN 'Electronics'
            ELSE 'Other'
          END AS Category
        FROM SalesFact;
        '''
        conn.executescript(category_sql)
        sql = '''
        SELECT pc.Category, ROUND(SUM(sf.TotalSales),2) AS total_sales
        FROM SalesFact sf
        JOIN ProductCategory pc ON sf.StockCode = pc.StockCode
        GROUP BY pc.Category
        ORDER BY total_sales DESC;
        '''
        return pd.read_sql(sql, conn)
