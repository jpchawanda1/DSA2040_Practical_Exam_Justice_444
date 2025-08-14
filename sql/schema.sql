-- Star schema DDL (SQLite)
DROP TABLE IF EXISTS main.SalesFact;
DROP TABLE IF EXISTS main.CustomerDim;
DROP TABLE IF EXISTS main.ProductDim;
DROP TABLE IF EXISTS main.TimeDim;

CREATE TABLE main.CustomerDim (
    CustomerKey INTEGER PRIMARY KEY,
    CustomerIDOriginal INTEGER,
    Country TEXT,
    customer_total_quantity INTEGER,
    customer_total_sales REAL,
    invoice_count INTEGER,
    first_purchase_date TEXT,
    last_purchase_date TEXT
);

CREATE TABLE main.ProductDim (
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

CREATE TABLE main.TimeDim (
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

CREATE TABLE main.SalesFact (
    FactID INTEGER PRIMARY KEY,
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

CREATE INDEX IF NOT EXISTS idx_customer_country ON main.CustomerDim(Country);
CREATE INDEX IF NOT EXISTS idx_product_category ON main.ProductDim(Category);
CREATE INDEX IF NOT EXISTS idx_time_year_month ON main.TimeDim(Year, Month);
CREATE INDEX IF NOT EXISTS idx_fact_date ON main.SalesFact(DateKey);
CREATE INDEX IF NOT EXISTS idx_fact_customer ON main.SalesFact(CustomerKey);
CREATE INDEX IF NOT EXISTS idx_fact_product ON main.SalesFact(ProductKey);
