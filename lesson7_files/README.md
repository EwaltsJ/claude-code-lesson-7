# E-Commerce Sales Analysis

## Contents

- `EDA_refactored.ipynb` - the analysis notebook: business objectives, data dictionary, configuration, data preparation, metric calculations, charts, and a summary section.
- `data_loader.py` - loads the raw CSV files and prepares the order-line sales table (joins, delivered-order filter, period columns, delivery days, category/state/review lookups).
- `business_metrics.py` - metric calculations only (revenue, growth, average order value, order counts, revenue by category/state, delivery speed, review scores, order status).
- `requirements.txt` - Python dependencies for the notebook and modules.
- `ecommerce_data/` - source CSV files (orders, order_items, products, customers, order_reviews).

## Setup

```bash
pip install -r requirements.txt
jupyter lab EDA_refactored.ipynb
```

## Running the analysis

Open `EDA_refactored.ipynb` and run all cells top to bottom. Every metric and chart in the notebook is computed from the `CONFIG` dictionary defined in the Configuration section:

```python
CONFIG = {
    "data_dir": Path("ecommerce_data"),
    "current_year": 2023,
    "current_month": None,   # None analyzes the full year; set 1-12 to analyze a single month
    "comparison_year": 2022,
    "comparison_month": None,
    "delivery_speed_bins": [0, 3, 7, float("inf")],
    "delivery_speed_labels": ["1-3 days", "4-7 days", "8+ days"],
}
```

To analyze a different period, change `current_year` / `current_month` and `comparison_year` / `comparison_month`, then re-run the notebook. For example, to analyze March 2023 against March 2022:

```python
CONFIG["current_year"] = 2023
CONFIG["current_month"] = 3
CONFIG["comparison_year"] = 2022
CONFIG["comparison_month"] = 3
```

Delivery-speed buckets used in the customer experience section are controlled the same way, through `delivery_speed_bins` and `delivery_speed_labels`. Bin edges follow the `pandas.cut` convention: `[0, 3, 7, float("inf")]` with three labels produces the buckets `(0, 3]`, `(3, 7]`, `(7, inf]`.

## Reusing the modules elsewhere

`data_loader.py` and `business_metrics.py` have no dependency on the notebook and can be imported directly in other scripts or notebooks that follow the same data schema (see the data dictionary in the notebook):

```python
import data_loader as dl
import business_metrics as bm

raw = dl.load_datasets("ecommerce_data")
sales = dl.build_order_line_sales(raw["orders"], raw["order_items"])
sales_delivered = dl.filter_delivered_orders(sales)

period_sales = dl.filter_period(sales_delivered, year=2024, month=6)
revenue = bm.calculate_total_revenue(period_sales)
```

Every function in both modules has a docstring describing its parameters and return value.
