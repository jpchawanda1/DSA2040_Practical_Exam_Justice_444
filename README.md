# DSA 2040 Practical Exam – Submission Overview

This submission includes a complete retail data warehouse build (ETL + OLAP) and a separate data mining component (preprocessing, clustering, classification, association rules). It uses the provided Online Retail Excel file for the warehouse and standard ML datasets/transactions for the data mining tasks. All required deliverables (.py scripts, .sql files, schema diagram, reports, and artifacts) are included.

## 1) Overview of the Submission

- Data Warehouse (DW):
  - Star schema implemented in SQLite with a single transactional fact and three dimensions.
  - ETL from `raw_data/Online Retail.xlsx` into `data_warehouse_notebook/retail_dw.db`.
  - OLAP queries and one visualization demonstrating roll-up, drill-down, and slice operations.
  - SQL deliverables in `sql/` and a lightweight star schema diagram under `docs/`.

- Data Mining (DM):
  - Task 1: Data preprocessing and exploration.
  - Task 2: KMeans clustering with metrics and visualizations.
  - Task 3: Classification (Decision Tree + KNN) and Association Rule Mining (Apriori or fallback).

Key notebooks are listed in Project Structure below.

## 2) Datasets Used

- Online Retail (UCI) – provided Excel file at `raw_data/Online Retail.xlsx`.
  - Used for the DW ETL to build Customer, Product, and Time dimensions and the Sales fact.
- Iris dataset (from scikit-learn) – used in data mining preprocessing, clustering, and classification tasks.
- Synthetic transactional basket data – used only for association rule mining where a retail transaction dataset is not available in the repo; Apriori is run via `mlxtend` if installed, with a simple pairwise fallback if not.

## 3) Implemented Star Schema (as built)

Grain: One row per invoice line (line-item at time of sale).

Tables created in `retail_dw.db`:
- SalesFact(FactID, InvoiceNo, ProductKey, CustomerKey, DateKey, Quantity, UnitPrice, TotalSales, Country)
- CustomerDim(CustomerKey, CustomerIDOriginal, Country, customer_total_quantity, customer_total_sales, invoice_count, first_purchase_date, last_purchase_date)
- ProductDim(ProductKey, StockCode, Description, Category, product_total_quantity, product_total_sales, distinct_invoices, first_sale_date, last_sale_date)
- TimeDim(DateKey, Date, Year, Quarter, Month, MonthName, Day, DayOfWeek, IsWeekend)

ASCII schema (see also `docs/schema_star.svg`):
```
       +-----------------+          +------------------+
       |   CustomerDim   |          |     TimeDim      |
       |-----------------|          |------------------|
       | CustomerKey (PK)|          | DateKey (PK)     |
       | CustomerIDOrig  |          | Date, Y,Q,M,...  |
       | Country         |          +---------^--------+
       +---------^-------+                    |
                 |                            |
       +---------+-------+          +---------+--------+
       |  ProductDim     |          |     SalesFact    |
       |-----------------|          |------------------|
       | ProductKey (PK) |<---------| ProductKey (FK)  |
       | StockCode       |    +-----| CustomerKey (FK) |
       | Description     |    |     | DateKey (FK)     |
       | Category        |    |     | InvoiceNo        |
       +-----------------+    |     | Quantity         |
                               |     | UnitPrice        |
                               |     | TotalSales       |
                               |     | Country          |
                               |     +------------------+
                               +----- CustomerDim ----->
                                      TimeDim --------->
```

Notes:
- We keep Country on the fact for convenience and also capture dominant country at the customer level. No separate GeographyDim in this build.
- Product category is inferred via simple keywords; it can be replaced by a governed product master later.

Why star schema over snowflake? Star keeps joins minimal and queries simple/fast for OLAP; denormalization (e.g., Category on ProductDim) trades a small amount of storage for better query performance and ease of use.

## 4) Project Structure (key items)

- `data_warehouse_notebook/etl_task2.ipynb` – ETL: reads Excel, cleans, builds dims, loads fact, writes `retail_dw.db`.
- `data_warehouse_notebook/task3_olap_analysis.ipynb` – OLAP queries (roll-up, drill-down, slice) + visualization.
- `data_warehouse_notebook/retail_dw.db` – SQLite database produced by ETL.
- `data_mining_notebook/task1_data_preprocessing.ipynb` – preprocessing, scaling/encoding, EDA plots.
- `data_mining_notebook/task2_clustering.ipynb` – KMeans clustering, metrics, elbow curve, PCA plot.
- `data_mining_notebook/task3_classification_association.ipynb` – Decision Tree & KNN + association rules.
- `data_mining_notebook/task3b_association_rules.ipynb` – stand-alone association rules focus.
- `data_*/*/artifacts/` – generated figures, CSVs, and metrics JSONs.

## 5) How to Run (Windows cmd)

Set up a virtual environment and required packages:

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend openpyxl jupyter
```

Run the ETL and OLAP notebooks or scripts:
- Open `data_warehouse_notebook/etl_task2.ipynb` and Run All to generate `retail_dw.db`.
- Then open `data_warehouse_notebook/task3_olap_analysis.ipynb` and Run All to execute OLAP queries and save the country chart to `data_warehouse_notebook/artifacts/`.

Alternatively, run the scripts:

```cmd
# Data Warehousing
python scripts\etl_retail.py

# Data Mining
python scripts\preprocessing_iris.py
python scripts\clustering_iris.py
python scripts\mining_iris_basket.py
```

Run the Data Mining notebooks similarly in `data_mining_notebook/` (Run All). If `mlxtend` is not installed, the notebooks will fall back to a simple pairwise association rules approximation.

## 6) Deliverables and Artifacts

- SQLite database: `data_warehouse_notebook/retail_dw.db`
- OLAP figure(s): `data_warehouse_notebook/artifacts/fig_task3_sales_by_country.png`
- Data mining artifacts: multiple PNGs/CSVs/JSONs under `data_mining_notebook/artifacts/` (e.g., elbow curve, PCA plot, decision tree, top rules CSV, metrics JSON).

Additional deliverables:
- `sql/schema.sql` – full DDL for the star schema (SQLite syntax).
- `sql/olap_queries.sql` – three OLAP queries (roll-up, drill-down, slice).
- `docs/schema_star.svg` – star schema diagram.
- `reports/olap_analysis.md` – 200–300 word OLAP analysis.

## 7) Mapping to Exam Tasks

- Section 1 – Data Warehousing
  - Task 1 (Design): `docs/schema_star.svg` (diagram), README (schema explanation + star vs snowflake), `sql/schema.sql` (CREATE TABLEs)
  - Task 2 (ETL): `scripts/etl_retail.py` (runs ETL), implementation in `utils/etl.py`, output DB at `data_warehouse_notebook/retail_dw.db`
  - Task 3 (OLAP): `sql/olap_queries.sql` (3 queries), visualization saved by OLAP notebook, analysis in `reports/olap_analysis.md`
- Section 2 – Data Mining
  - Task 1 (Preprocessing/EDA): `scripts/preprocessing_iris.py` (+ artifacts)
  - Task 2 (Clustering): `scripts/clustering_iris.py` (+ artifacts)
  - Task 3 (Classification + Association): `scripts/mining_iris_basket.py` (+ artifacts)

## 7) Self‑Assessment (what’s done vs pending)

Completed:
- ETL pipeline from Excel with cleaning, outlier handling, and robust date window fallback.
- Star schema build with CustomerDim, ProductDim (newly added), TimeDim, and SalesFact referencing ProductKey.
- Indexes and sanity tests (foreign key coverage, TotalSales consistency, simple analytical checks).
- OLAP notebook with roll‑up, drill‑down, and category slice; visualization saved to artifacts.
- Data mining notebooks for preprocessing, clustering, classification, and association rules (with fallback).

Limitations / Next steps:
- No separate GeographyDim; Country is kept on fact and dominant country tracked on CustomerDim. If needed, add a GeographyDim and refactor queries.
- Product Category is inferred via keywords; replace with a governed product/category lookup for accuracy.
- Association rules use synthetic transactions due to the lack of basket-level retail data in the repo; integrate a real basket dataset for stronger results when available.
- Documentation diagrams are concise; a formal ERD could be added.