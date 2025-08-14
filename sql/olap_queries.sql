-- Roll-up: Total sales by country and quarter
SELECT sf.Country, t.Year, t.Quarter, ROUND(SUM(sf.TotalSales),2) AS total_sales
FROM SalesFact sf
JOIN TimeDim t ON sf.DateKey = t.DateKey
GROUP BY sf.Country, t.Year, t.Quarter
ORDER BY total_sales DESC;

-- Drill-down: Monthly sales for a specific country (parameterize :country)
SELECT t.Year, t.Month, SUM(sf.TotalSales) AS monthly_sales
FROM SalesFact sf
JOIN TimeDim t ON sf.DateKey = t.DateKey
WHERE sf.Country = :country
GROUP BY t.Year, t.Month
ORDER BY t.Year, t.Month;

-- Slice: Total sales by product category (requires ProductDim)
SELECT p.Category AS category, ROUND(SUM(f.TotalSales),2) AS total_sales
FROM SalesFact f
JOIN ProductDim p ON f.ProductKey = p.ProductKey
GROUP BY p.Category
ORDER BY total_sales DESC;
