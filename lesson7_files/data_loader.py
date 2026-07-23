"""Data loading and preparation utilities for the e-commerce sales analysis.

This module is responsible for everything that happens before business
metrics are calculated: reading the raw CSV files, joining them into an
order-line-level sales table, and deriving the columns (year, month,
delivery time) that the metric functions in business_metrics.py depend on.

No business metrics are calculated here.
"""

from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path("ecommerce_data")

DELIVERED_STATUS = "delivered"


def load_datasets(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Read the raw e-commerce CSV files into a dictionary of DataFrames.

    Parameters
    ----------
    data_dir : Path
        Directory containing the source CSV files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: 'orders', 'order_items', 'products', 'customers', 'reviews'.
    """
    data_dir = Path(data_dir)
    return {
        "orders": pd.read_csv(data_dir / "orders_dataset.csv"),
        "order_items": pd.read_csv(data_dir / "order_items_dataset.csv"),
        "products": pd.read_csv(data_dir / "products_dataset.csv"),
        "customers": pd.read_csv(data_dir / "customers_dataset.csv"),
        "reviews": pd.read_csv(data_dir / "order_reviews_dataset.csv"),
    }


def add_period_columns(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Parse a timestamp column and derive 'year' and 'month' columns from it.

    Used both for the order-line sales table and the raw orders table, so
    any date-stamped table can be filtered to a calendar period with
    filter_period.

    Parameters
    ----------
    df : pd.DataFrame
        Table containing a timestamp column.
    date_column : str
        Name of the column to parse and derive year/month from.

    Returns
    -------
    pd.DataFrame
        Copy of the input with the date column parsed to datetime and
        'year' and 'month' columns added.
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df["year"] = df[date_column].dt.year
    df["month"] = df[date_column].dt.month
    return df


def build_order_line_sales(orders: pd.DataFrame, order_items: pd.DataFrame) -> pd.DataFrame:
    """Join order items with order-level fields into one order-line sales table.

    Adds parsed timestamps plus 'year' and 'month' columns derived from
    order_purchase_timestamp, so the resulting table can be filtered to any
    calendar period downstream.

    Parameters
    ----------
    orders : pd.DataFrame
        Raw orders dataset.
    order_items : pd.DataFrame
        Raw order items dataset.

    Returns
    -------
    pd.DataFrame
        One row per order line, with order_status, purchase/delivery
        timestamps, customer_id, year and month.
    """
    sales = pd.merge(
        left=order_items[["order_id", "order_item_id", "product_id", "price"]],
        right=orders[
            [
                "order_id",
                "customer_id",
                "order_status",
                "order_purchase_timestamp",
                "order_delivered_customer_date",
            ]
        ],
        on="order_id",
        how="inner",
    )

    sales["order_delivered_customer_date"] = pd.to_datetime(sales["order_delivered_customer_date"])
    sales = add_period_columns(sales, "order_purchase_timestamp")

    return sales


def filter_delivered_orders(sales: pd.DataFrame) -> pd.DataFrame:
    """Return only order lines belonging to delivered orders.

    Parameters
    ----------
    sales : pd.DataFrame
        Order-line sales table containing an 'order_status' column.

    Returns
    -------
    pd.DataFrame
        Subset of rows where order_status == 'delivered'.
    """
    return sales[sales["order_status"] == DELIVERED_STATUS].copy()


def filter_period(df: pd.DataFrame, year: int, month: int | None = None) -> pd.DataFrame:
    """Filter a table to a configurable calendar period.

    Works on any table that has 'year' and 'month' columns, such as those
    produced by add_period_columns or build_order_line_sales. This is the
    single place that defines what a "period" means, so the same function
    supports a full year, a single month, or (by calling it twice) a
    custom range.

    Parameters
    ----------
    df : pd.DataFrame
        Table with 'year' and 'month' columns.
    year : int
        Calendar year to filter to.
    month : int, optional
        Calendar month (1-12) to filter to. If None, the entire year is returned.

    Returns
    -------
    pd.DataFrame
        Rows matching the requested year (and month, if provided).
    """
    period_df = df[df["year"] == year]
    if month is not None:
        period_df = period_df[period_df["month"] == month]
    return period_df.copy()


def add_delivery_days(sales: pd.DataFrame) -> pd.DataFrame:
    """Add a 'delivery_days' column: days between purchase and customer delivery.

    Parameters
    ----------
    sales : pd.DataFrame
        Sales table with 'order_purchase_timestamp' and
        'order_delivered_customer_date' as datetime columns.

    Returns
    -------
    pd.DataFrame
        Copy of the input with a 'delivery_days' column added. Orders that
        have not been delivered yet will have a null value.
    """
    sales = sales.copy()
    sales["delivery_days"] = (
        sales["order_delivered_customer_date"] - sales["order_purchase_timestamp"]
    ).dt.days
    return sales


def attach_product_category(sales: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Add product_category_name to a sales table by joining on product_id.

    Parameters
    ----------
    sales : pd.DataFrame
        Sales table with a 'product_id' column.
    products : pd.DataFrame
        Raw products dataset.

    Returns
    -------
    pd.DataFrame
        Sales table with 'product_category_name' added.
    """
    return pd.merge(
        left=sales,
        right=products[["product_id", "product_category_name"]],
        on="product_id",
        how="left",
    )


def attach_customer_state(sales: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Add customer_state to a sales table by joining on customer_id.

    Parameters
    ----------
    sales : pd.DataFrame
        Sales table with a 'customer_id' column.
    customers : pd.DataFrame
        Raw customers dataset.

    Returns
    -------
    pd.DataFrame
        Sales table with 'customer_state' added.
    """
    return pd.merge(
        left=sales,
        right=customers[["customer_id", "customer_state"]],
        on="customer_id",
        how="left",
    )


def get_order_level_view(sales: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Collapse an order-line sales table to one row per order.

    Order-level attributes (e.g. delivery_days, review_score) are repeated
    across every line item of an order; this removes the duplication before
    those attributes are averaged or grouped.

    Parameters
    ----------
    sales : pd.DataFrame
        Order-line sales table.
    columns : list[str]
        Columns to keep, must include the order identifier.

    Returns
    -------
    pd.DataFrame
        One row per unique combination of the requested columns.
    """
    return sales[columns].drop_duplicates()


def attach_review_score(sales: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Add review_score to a sales table by joining on order_id.

    Orders with more than one review keep only the first match, and orders
    without a review are dropped, matching an inner join semantic.

    Parameters
    ----------
    sales : pd.DataFrame
        Sales table with an 'order_id' column.
    reviews : pd.DataFrame
        Raw order reviews dataset.

    Returns
    -------
    pd.DataFrame
        Sales table with 'review_score' added.
    """
    return pd.merge(
        left=sales,
        right=reviews[["order_id", "review_score"]].drop_duplicates(subset="order_id"),
        on="order_id",
        how="inner",
    )
