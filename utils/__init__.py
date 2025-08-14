"""Utility package for ETL, OLAP, and data mining helpers.

Modules:
- etl: Functions to read, clean, transform, and load the Online Retail dataset into SQLite star schema.
- olap: Helper queries and plotting utilities on top of the star schema.
- dm: (optional) Shared utilities for data mining notebooks.
"""

__all__ = [
    'etl',
    'olap',
    'dm',
]
