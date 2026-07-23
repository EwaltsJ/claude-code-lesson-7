"""Business metric calculations for the e-commerce sales analysis.

Every function here takes an already-prepared DataFrame (see data_loader.py)
and returns a metric, a comparison, or a table ready for plotting. Functions
are period-agnostic: callers decide which rows belong to which period
before calling these functions, so the same code works for a single month,
a full year, or any other range.
"""

import pandas as pd

PRICE_COLUMN = "price"
ORDER_ID_COLUMN = "order_id"


def calculate_total_revenue(sales: pd.DataFrame, price_column: str = PRICE_COLUMN) -> float:
    """Sum the price column of a sales table.

    Parameters
    ----------
    sales : pd.DataFrame
        Order-line sales table for the period of interest.
    price_column : str
        Name of the column holding the line-item price.

    Returns
    -------
    float
        Total revenue for the rows provided.
    """
    return sales[price_column].sum()


def calculate_order_count(sales: pd.DataFrame, order_id_column: str = ORDER_ID_COLUMN) -> int:
    """Count the number of distinct orders in a sales table.

    Parameters
    ----------
    sales : pd.DataFrame
        Order-line sales table for the period of interest.
    order_id_column : str
        Name of the column holding the order identifier.

    Returns
    -------
    int
        Number of unique orders.
    """
    return sales[order_id_column].nunique()


def calculate_average_order_value(
    sales: pd.DataFrame,
    order_id_column: str = ORDER_ID_COLUMN,
    price_column: str = PRICE_COLUMN,
) -> float:
    """Calculate the average revenue per order.

    Line items are summed per order before averaging, so a multi-item order
    contributes its full order value rather than being counted per line.

    Parameters
    ----------
    sales : pd.DataFrame
        Order-line sales table for the period of interest.
    order_id_column : str
        Name of the column holding the order identifier.
    price_column : str
        Name of the column holding the line-item price.

    Returns
    -------
    float
        Mean order value across the period.
    """
    order_totals = sales.groupby(order_id_column)[price_column].sum()
    return order_totals.mean()


def calculate_percent_change(current_value: float, previous_value: float) -> float:
    """Calculate the percentage change between two values.

    Parameters
    ----------
    current_value : float
        Value for the more recent period.
    previous_value : float
        Value for the comparison period.

    Returns
    -------
    float
        Percentage change from previous_value to current_value. Returns NaN
        if previous_value is zero, since the change is undefined.
    """
    if previous_value == 0:
        return float("nan")
    return (current_value - previous_value) / previous_value * 100


def calculate_monthly_revenue(
    sales: pd.DataFrame,
    month_column: str = "month",
    price_column: str = PRICE_COLUMN,
) -> pd.Series:
    """Calculate total revenue for each month present in a sales table.

    Parameters
    ----------
    sales : pd.DataFrame
        Order-line sales table, typically pre-filtered to a single year.
    month_column : str
        Name of the column holding the calendar month (1-12).
    price_column : str
        Name of the column holding the line-item price.

    Returns
    -------
    pd.Series
        Revenue indexed by month, sorted chronologically.
    """
    return sales.groupby(month_column)[price_column].sum().sort_index()


def calculate_month_over_month_growth(monthly_revenue: pd.Series) -> pd.Series:
    """Calculate the percentage change in revenue from each month to the next.

    Parameters
    ----------
    monthly_revenue : pd.Series
        Revenue indexed by month, as returned by calculate_monthly_revenue.

    Returns
    -------
    pd.Series
        Percentage change between consecutive months. The first month is
        NaN, since there is no prior month to compare against.
    """
    return monthly_revenue.pct_change() * 100


def calculate_revenue_by_dimension(
    sales: pd.DataFrame,
    dimension_column: str,
    price_column: str = PRICE_COLUMN,
) -> pd.Series:
    """Calculate total revenue grouped by an arbitrary categorical column.

    Used for both product-category and geographic breakdowns; any
    categorical column (e.g. product_category_name, customer_state) can be
    passed in.

    Parameters
    ----------
    sales : pd.DataFrame
        Order-line sales table for the period of interest.
    dimension_column : str
        Name of the categorical column to group by.
    price_column : str
        Name of the column holding the line-item price.

    Returns
    -------
    pd.Series
        Revenue per category, sorted from highest to lowest.
    """
    return (
        sales.groupby(dimension_column)[price_column]
        .sum()
        .sort_values(ascending=False)
    )


def calculate_order_status_distribution(orders: pd.DataFrame, status_column: str = "order_status") -> pd.Series:
    """Calculate the share of orders in each order_status category.

    Parameters
    ----------
    orders : pd.DataFrame
        Orders table (one row per order), typically pre-filtered to a period.
    status_column : str
        Name of the column holding the order status.

    Returns
    -------
    pd.Series
        Proportion of orders in each status, summing to 1.
    """
    return orders[status_column].value_counts(normalize=True)


def calculate_average_delivery_days(sales: pd.DataFrame, delivery_days_column: str = "delivery_days") -> float:
    """Calculate the average number of days between purchase and delivery.

    Parameters
    ----------
    sales : pd.DataFrame
        Sales table with a 'delivery_days' column (see data_loader.add_delivery_days).
    delivery_days_column : str
        Name of the column holding delivery time in days.

    Returns
    -------
    float
        Mean delivery time in days.
    """
    return sales[delivery_days_column].mean()


def categorize_delivery_speed(
    sales: pd.DataFrame,
    bin_edges: list[float],
    bin_labels: list[str],
    delivery_days_column: str = "delivery_days",
    output_column: str = "delivery_speed_bucket",
) -> pd.DataFrame:
    """Bucket orders into delivery-speed categories using caller-supplied bins.

    Bin edges and labels are passed in by the caller rather than assumed,
    so the same function supports any speed grouping the analyst chooses.

    Parameters
    ----------
    sales : pd.DataFrame
        Sales table with a delivery-days column.
    bin_edges : list[float]
        Bin edges passed to pandas.cut, e.g. [0, 3, 7, float('inf')].
    bin_labels : list[str]
        One label per bin, e.g. ['1-3 days', '4-7 days', '8+ days'].
        Must have length len(bin_edges) - 1.
    delivery_days_column : str
        Name of the column holding delivery time in days.
    output_column : str
        Name of the new column holding the assigned bucket label.

    Returns
    -------
    pd.DataFrame
        Copy of the input with the bucket column added.
    """
    sales = sales.copy()
    sales[output_column] = pd.cut(
        sales[delivery_days_column], bins=bin_edges, labels=bin_labels
    )
    return sales


def calculate_average_review_score_by_group(
    sales: pd.DataFrame,
    group_column: str,
    review_score_column: str = "review_score",
) -> pd.DataFrame:
    """Calculate mean review score for each value of a grouping column.

    Used both for review score by exact delivery day and by delivery-speed
    bucket; any grouping column can be supplied.

    Parameters
    ----------
    sales : pd.DataFrame
        Sales table with review_score and the grouping column.
    group_column : str
        Name of the column to group by (e.g. 'delivery_days', 'delivery_speed_bucket').
    review_score_column : str
        Name of the column holding the review score.

    Returns
    -------
    pd.DataFrame
        Two columns: group_column and mean review_score_column.
    """
    return (
        sales.groupby(group_column, observed=True)[review_score_column]
        .mean()
        .reset_index()
    )


def calculate_average_review_score(sales: pd.DataFrame, review_score_column: str = "review_score") -> float:
    """Calculate the overall average review score.

    Parameters
    ----------
    sales : pd.DataFrame
        Sales table with a review_score column.
    review_score_column : str
        Name of the column holding the review score.

    Returns
    -------
    float
        Mean review score across all rows.
    """
    return sales[review_score_column].mean()


def calculate_review_score_distribution(sales: pd.DataFrame, review_score_column: str = "review_score") -> pd.Series:
    """Calculate the share of reviews at each score value.

    Parameters
    ----------
    sales : pd.DataFrame
        Sales table with a review_score column.
    review_score_column : str
        Name of the column holding the review score.

    Returns
    -------
    pd.Series
        Proportion of reviews at each score, indexed by score.
    """
    return sales[review_score_column].value_counts(normalize=True).sort_index()
