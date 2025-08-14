from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import sqlite3

logger = logging.getLogger("utils.etl")


# ------------------------
# Parameters and constants
# ------------------------
DEFAULT_EXCEL_FILENAME = "Online Retail.xlsx"


# ------------------------
# Extract & Clean
# ------------------------
def load_excel(data_dir: Path, excel_filename: str = DEFAULT_EXCEL_FILENAME) -> pd.DataFrame:
    path = (data_dir / excel_filename).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Expected Excel file at {path}")
    try:
        df = pd.read_excel(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read Excel: {e}")
    return df


def clean_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    required_cols = ['InvoiceNo', 'StockCode', 'Quantity', 'UnitPrice', 'InvoiceDate']
    if 'CustomerID' in df.columns:
        df['CustomerID'] = df['CustomerID'].fillna(-1)

    df = df.dropna(subset=required_cols)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate'])

    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0).astype(int)
    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce').fillna(0.0).astype(float)

    for col in ['InvoiceNo', 'StockCode', 'Description', 'Country']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Remove outliers and add TotalSales
    outlier_mask = (df['Quantity'] < 0) | (df['UnitPrice'] <= 0)
    df = df.loc[~outlier_mask].copy()
    df['TotalSales'] = df['Quantity'] * df['UnitPrice']
    return df


def select_last_year_window(df: pd.DataFrame, fixed_current_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    current_date = fixed_current_date
    last_year_start = current_date - pd.Timedelta(days=365)

    max_date_in_data = df['InvoiceDate'].max()
    if pd.notna(max_date_in_data) and max_date_in_data < current_date - pd.Timedelta(days=30):
        current_date = max_date_in_data.normalize()
        last_year_start = current_date - pd.Timedelta(days=365)

    df_last_year = df[(df['InvoiceDate'] >= last_year_start) & (df['InvoiceDate'] <= current_date)].copy()
    if df_last_year.empty:
        df_last_year = df.copy()
        last_year_start = df_last_year['InvoiceDate'].min().normalize()

    return df_last_year, current_date, last_year_start


# ------------------------
# Dimensions
# ------------------------
def build_customer_dim(df_last_year: pd.DataFrame) -> pd.DataFrame:
    customer_group = df_last_year.groupby('CustomerID', dropna=False)
    customer_dim = customer_group.agg(
        customer_total_quantity=('Quantity', 'sum'),
        customer_total_sales=('TotalSales', 'sum'),
        invoice_count=('InvoiceNo', 'nunique'),
        first_purchase_date=('InvoiceDate', 'min'),
        last_purchase_date=('InvoiceDate', 'max'),
        country=('Country', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
    ).reset_index().rename(columns={'CustomerID': 'CustomerIDOriginal'})

    customer_dim = customer_dim[customer_dim['CustomerIDOriginal'] != -1].copy()
    customer_dim.insert(0, 'CustomerKey', range(1, len(customer_dim) + 1))
    return customer_dim


def build_product_dim(df_last_year: pd.DataFrame) -> pd.DataFrame:
    product_group = df_last_year.groupby(['StockCode', 'Description'], dropna=False)
    product_dim = product_group.agg(
        product_total_quantity=('Quantity', 'sum'),
        product_total_sales=('TotalSales', 'sum'),
        distinct_invoices=('InvoiceNo', 'nunique'),
        first_sale_date=('InvoiceDate', 'min'),
        last_sale_date=('InvoiceDate', 'max')
    ).reset_index()

    kw = ['LED', 'LIGHT', 'LAMP', 'ELECTRIC', 'BATTERY']
    upper_desc = product_dim['Description'].fillna('').str.upper()
    product_dim['Category'] = np.where(
        upper_desc.str.contains('|'.join(kw)), 'Electronics', 'Other'
    )

    product_dim.insert(0, 'ProductKey', range(1, len(product_dim) + 1))
    return product_dim


def build_time_dim(df_last_year: pd.DataFrame) -> pd.DataFrame:
    unique_dates = pd.to_datetime(df_last_year['InvoiceDate'].dt.normalize().unique())
    time_dim = pd.DataFrame({'Date': unique_dates})
    time_dim['DateKey'] = time_dim['Date'].dt.strftime('%Y%m%d').astype(int)
    time_dim['Year'] = time_dim['Date'].dt.year
    time_dim['Quarter'] = time_dim['Date'].dt.quarter
    time_dim['Month'] = time_dim['Date'].dt.month
    time_dim['MonthName'] = time_dim['Date'].dt.month_name().str[:3]
    time_dim['Day'] = time_dim['Date'].dt.day
    time_dim['DayOfWeek'] = time_dim['Date'].dt.dayofweek + 1
    time_dim['IsWeekend'] = time_dim['DayOfWeek'].isin([6, 7]).astype(int)
    return time_dim


# ------------------------
# Fact
# ------------------------
def build_fact(df_last_year: pd.DataFrame, customer_dim: pd.DataFrame, product_dim: pd.DataFrame, time_dim: pd.DataFrame) -> pd.DataFrame:
    cust_map = dict(zip(customer_dim['CustomerIDOriginal'], customer_dim['CustomerKey']))
    df_fact = df_last_year.copy()
    df_fact['CustomerKey'] = df_fact['CustomerID'].map(cust_map)

    df_fact = df_fact.merge(
        product_dim[['ProductKey', 'StockCode', 'Description']],
        on=['StockCode', 'Description'],
        how='left'
    )

    norm_dates = df_fact['InvoiceDate'].dt.normalize()
    DateKey_map = dict(zip(time_dim['Date'], time_dim['DateKey']))
    df_fact['DateKey'] = norm_dates.map(DateKey_map)

    fact_cols = ['InvoiceNo', 'ProductKey', 'CustomerKey', 'DateKey', 'Quantity', 'UnitPrice', 'TotalSales', 'Country']
    fact_df = df_fact[fact_cols].dropna(subset=['CustomerKey', 'DateKey', 'ProductKey'])
    return fact_df


# ------------------------
# Load / DDL
# ------------------------
DDL_STATEMENTS = [
    "DROP TABLE IF EXISTS SalesFact;",
    "DROP TABLE IF EXISTS CustomerDim;",
    "DROP TABLE IF EXISTS ProductDim;",
    "DROP TABLE IF EXISTS TimeDim;",
    """
    CREATE TABLE CustomerDim (
        CustomerKey INTEGER PRIMARY KEY,
        CustomerIDOriginal INTEGER,
        Country TEXT,
        customer_total_quantity INTEGER,
        customer_total_sales REAL,
        invoice_count INTEGER,
        first_purchase_date TEXT,
        last_purchase_date TEXT
    );
    """,
    """
    CREATE TABLE ProductDim (
        ProductKey INTEGER PRIMARY KEY,
        StockCode TEXT,
        Description TEXT,
        Category TEXT,
        product_total_quantity INTEGER,
        product_total_sales REAL,
        distinct_invoices INTEGER,
        first_sale_date TEXT,
        last_sale_date TEXT
    );
    """,
    """
    CREATE TABLE TimeDim (
        DateKey INTEGER PRIMARY KEY,
        Date TEXT,
        Year INTEGER,
        Quarter INTEGER,
        Month INTEGER,
        MonthName TEXT,
        Day INTEGER,
        DayOfWeek INTEGER,
        IsWeekend INTEGER
    );
    """,
    """
    CREATE TABLE SalesFact (
        FactID INTEGER PRIMARY KEY AUTOINCREMENT,
        InvoiceNo TEXT,
        ProductKey INTEGER,
        CustomerKey INTEGER,
        DateKey INTEGER,
        Quantity INTEGER,
        UnitPrice REAL,
        TotalSales REAL,
        Country TEXT,
        FOREIGN KEY (CustomerKey) REFERENCES CustomerDim(CustomerKey),
        FOREIGN KEY (DateKey) REFERENCES TimeDim(DateKey),
        FOREIGN KEY (ProductKey) REFERENCES ProductDim(ProductKey)
    );
    """,
]


def recreate_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for stmt in DDL_STATEMENTS:
        cur.executescript(stmt)
    conn.commit()


def load_dimensions(conn: sqlite3.Connection, customer_dim: pd.DataFrame, product_dim: pd.DataFrame, time_dim: pd.DataFrame) -> None:
    customer_dim.to_sql('CustomerDim', conn, if_exists='append', index=False)
    product_dim.to_sql('ProductDim', conn, if_exists='append', index=False)
    time_dim.to_sql('TimeDim', conn, if_exists='append', index=False)
    cur = conn.cursor()
    cur.execute('CREATE INDEX IF NOT EXISTS idx_customer_country ON CustomerDim(Country);')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_product_category ON ProductDim(Category);')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_time_year_month ON TimeDim(Year, Month);')
    conn.commit()


def load_fact(conn: sqlite3.Connection, fact_df: pd.DataFrame) -> None:
    fact_df.to_sql('SalesFact', conn, if_exists='append', index=False)
    cur = conn.cursor()
    cur.execute('CREATE INDEX IF NOT EXISTS idx_fact_date ON SalesFact(DateKey);')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_fact_customer ON SalesFact(CustomerKey);')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_fact_product ON SalesFact(ProductKey);')
    conn.commit()


# ------------------------
# Orchestrator
# ------------------------

def run_etl(
    excel_path: Path,
    db_path: Path,
    fixed_current_date: pd.Timestamp,
) -> Dict[str, int]:
    df_excel = load_excel(excel_path.parent, excel_path.name)
    raw_row_count = len(df_excel)

    df_clean = clean_raw(df_excel)
    df_last_year, current_date, last_year_start = select_last_year_window(df_clean, fixed_current_date)

    customer_dim = build_customer_dim(df_last_year)
    product_dim = build_product_dim(df_last_year)
    time_dim = build_time_dim(df_last_year)
    fact_df = build_fact(df_last_year, customer_dim, product_dim, time_dim)

    conn = sqlite3.connect(db_path)
    recreate_schema(conn)
    load_dimensions(conn, customer_dim, product_dim, time_dim)
    load_fact(conn, fact_df)

    counts = {
        'raw': raw_row_count,
        'cleaned': len(df_clean),
        'last_year': len(df_last_year),
        'customers': len(customer_dim),
        'products': len(product_dim),
        'dates': len(time_dim),
        'fact': len(fact_df)
    }
    logger.info("ETL complete: %s", counts)
    conn.close()
    return counts
