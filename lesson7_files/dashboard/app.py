"""Streamlit dashboard for the e-commerce sales analysis.

Reuses the data preparation and metric-calculation logic from
data_loader.py and business_metrics.py (in the parent lesson7_files
directory) so the dashboard and the notebook stay backed by the same
business logic. All period filtering here is driven by the date-range
picker in the header rather than the notebook's year/month CONFIG.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
LESSON_DIR = APP_DIR.parent
DATA_DIR = LESSON_DIR / "ecommerce_data"
sys.path.append(str(LESSON_DIR))

import business_metrics as bm  # noqa: E402
import data_loader as dl  # noqa: E402

DELIVERY_SPEED_BINS = [0, 3, 7, float("inf")]
DELIVERY_SPEED_LABELS = ["1-3 days", "4-7 days", "8+ days"]

# ---------------------------------------------------------------------------
# Palette (validated sequential blue ramp + status colors, see dataviz skill)
# ---------------------------------------------------------------------------
COLOR_SURFACE = "#fcfcfb"
COLOR_PAGE = "#f9f9f7"
COLOR_PRIMARY_INK = "#0b0b0b"
COLOR_SECONDARY_INK = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRIDLINE = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"
COLOR_GOOD = "#0ca30c"
COLOR_CRITICAL = "#d03b3b"
COLOR_ACCENT = "#2a78d6"

SEQ_BLUE_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQ_BLUE_SCALE = [[i / (len(SEQ_BLUE_STEPS) - 1), c] for i, c in enumerate(SEQ_BLUE_STEPS)]

CHART_CARD_HEIGHT = 480
CHART_FIG_HEIGHT = 430

st.set_page_config(page_title="E-Commerce Sales Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = dl.load_datasets(DATA_DIR)
    sales = dl.build_order_line_sales(raw["orders"], raw["order_items"])
    sales = dl.filter_delivered_orders(sales)
    sales = dl.add_delivery_days(sales)
    sales = dl.attach_product_category(sales, raw["products"])
    sales = dl.attach_customer_state(sales, raw["customers"])
    return sales, raw["reviews"]


def filter_date_range(sales: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    ts = sales["order_purchase_timestamp"]
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    return sales[(ts >= start_ts) & (ts < end_ts)].copy()


def get_previous_period(start: date, end: date) -> tuple[date, date]:
    period_length = (end - start).days + 1
    prev_end = start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=period_length - 1)
    return prev_start, prev_end


def compute_period_data(sales: pd.DataFrame, reviews: pd.DataFrame, start: date, end: date) -> dict:
    period_sales = filter_date_range(sales, start, end)
    period_sales_reviews = dl.attach_review_score(period_sales, reviews)
    order_level = dl.get_order_level_view(
        period_sales_reviews, ["order_id", "delivery_days", "review_score"]
    )
    return {
        "sales": period_sales,
        "order_level": order_level,
        "revenue": bm.calculate_total_revenue(period_sales),
        "aov": bm.calculate_average_order_value(period_sales),
        "order_count": bm.calculate_order_count(period_sales),
        "avg_review_score": bm.calculate_average_review_score(order_level),
        "avg_delivery_days": bm.calculate_average_delivery_days(order_level),
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def format_currency_abbrev(value: float) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000_000:
        number, suffix = value / 1_000_000_000, "B"
    elif value >= 1_000_000:
        number, suffix = value / 1_000_000, "M"
    elif value >= 1_000:
        number, suffix = value / 1_000, "K"
    else:
        return f"{sign}${value:.0f}"
    text = f"{number:.1f}".rstrip("0").rstrip(".")
    return f"{sign}${text}{suffix}"


def format_currency_full(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def format_delta(value: float) -> str:
    return f"{value:+.2f}%"


def nice_ticks(max_value: float, target_count: int = 5) -> list[float]:
    if not max_value or max_value <= 0 or pd.isna(max_value):
        return [0]
    raw_step = max_value / target_count
    magnitude = 10 ** np.floor(np.log10(raw_step))
    residual = raw_step / magnitude
    if residual > 5:
        step = 10 * magnitude
    elif residual > 2:
        step = 5 * magnitude
    elif residual > 1:
        step = 2 * magnitude
    else:
        step = magnitude
    return list(np.arange(0, max_value + step, step))


def trend_arrow_and_color(pct_change: float, good_direction: str = "up") -> tuple[str, str]:
    if pd.isna(pct_change):
        return "–", COLOR_MUTED
    if pct_change == 0:
        return "–", COLOR_MUTED
    is_up = pct_change > 0
    is_good = is_up if good_direction == "up" else not is_up
    arrow = "▲" if is_up else "▼"
    color = COLOR_GOOD if is_good else COLOR_CRITICAL
    return arrow, color


def render_stars(score: float, max_stars: int = 5) -> str:
    if pd.isna(score):
        filled = 0
    else:
        filled = max(0, min(max_stars, int(round(score))))
    empty = max_stars - filled
    filled_html = f'<span style="color:{COLOR_PRIMARY_INK};">{"★" * filled}</span>'
    empty_html = f'<span style="color:{COLOR_BASELINE};">{"★" * empty}</span>'
    return filled_html + empty_html


# ---------------------------------------------------------------------------
# Card renderers (HTML, fixed heights for row uniformity)
# ---------------------------------------------------------------------------
def kpi_card(title: str, value: str, delta_pct: float | None, good_direction: str, subtitle: str) -> str:
    delta_html = ""
    if delta_pct is not None:
        arrow, color = trend_arrow_and_color(delta_pct, good_direction)
        delta_html = f'<div class="kpi-delta" style="color:{color};">{arrow} {format_delta(delta_pct)}</div>'
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
        <div class="kpi-subtitle">{subtitle}</div>
    </div>
    """


def kpi_growth_card(title: str, pct: float, subtitle: str) -> str:
    arrow, color = trend_arrow_and_color(pct, "up")
    value_text = "N/A" if pd.isna(pct) else f"{pct:+.2f}%"
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value" style="color:{color};">{arrow} {value_text}</div>
        <div class="kpi-subtitle">{subtitle}</div>
    </div>
    """


def bottom_metric_card(title: str, value: str, delta_pct: float, good_direction: str, subtitle: str) -> str:
    arrow, color = trend_arrow_and_color(delta_pct, good_direction)
    delta_html = f'<div class="kpi-delta" style="color:{color};">{arrow} {format_delta(delta_pct)}</div>' if not pd.isna(delta_pct) else ""
    return f"""
    <div class="bottom-card">
        <div class="bottom-title">{title}</div>
        <div class="bottom-value">{value}</div>
        {delta_html}
        <div class="bottom-subtitle">{subtitle}</div>
    </div>
    """


def review_score_card(score: float) -> str:
    score_text = "N/A" if pd.isna(score) else f"{score:.2f}"
    return f"""
    <div class="bottom-card">
        <div class="bottom-value">{score_text}</div>
        <div class="bottom-stars">{render_stars(score)}</div>
        <div class="bottom-subtitle">Average Review Score</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def monthly_revenue_series(sales: pd.DataFrame) -> pd.Series:
    if sales.empty:
        return pd.Series(dtype=float)
    s = sales.copy()
    s["period_month"] = s["order_purchase_timestamp"].dt.to_period("M")
    return s.groupby("period_month")["price"].sum().sort_index()


def base_layout(title: str, height: int = CHART_FIG_HEIGHT) -> dict:
    return dict(
        height=height,
        margin=dict(l=10, r=20, t=45, b=10),
        plot_bgcolor=COLOR_SURFACE,
        paper_bgcolor=COLOR_SURFACE,
        title=dict(text=title, font=dict(size=15, color=COLOR_PRIMARY_INK)),
        font=dict(color=COLOR_SECONDARY_INK),
    )


def build_revenue_trend_chart(current_sales: pd.DataFrame, previous_sales: pd.DataFrame) -> go.Figure:
    cur_series = monthly_revenue_series(current_sales)
    prev_series = monthly_revenue_series(previous_sales)
    n = max(len(cur_series), len(prev_series), 1)
    x = list(range(1, n + 1))

    cur_y = list(cur_series.values) + [None] * (n - len(cur_series))
    prev_y = list(prev_series.values) + [None] * (n - len(prev_series))
    cur_labels = [p.strftime("%b %Y") for p in cur_series.index] + [""] * (n - len(cur_series))
    prev_labels = [p.strftime("%b %Y") for p in prev_series.index] + [""] * (n - len(prev_series))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=prev_y, mode="lines+markers", name="Previous period",
        line=dict(color=COLOR_MUTED, width=2, dash="dash"),
        marker=dict(size=6, color=COLOR_MUTED),
        customdata=prev_labels,
        hovertemplate="%{customdata}<br>Revenue: $%{y:,.0f}<extra>Previous period</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=cur_y, mode="lines+markers", name="Current period",
        line=dict(color=COLOR_ACCENT, width=2),
        marker=dict(size=7, color=COLOR_ACCENT),
        customdata=cur_labels,
        hovertemplate="%{customdata}<br>Revenue: $%{y:,.0f}<extra>Current period</extra>",
    ))

    all_values = [v for v in cur_y + prev_y if v is not None]
    ticks = nice_ticks(max(all_values, default=0))
    axis_labels = [c if c else p for c, p in zip(cur_labels, prev_labels)]

    fig.update_layout(**base_layout("Revenue Trend"))
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1),
        xaxis=dict(tickmode="array", tickvals=x, ticktext=axis_labels, showgrid=False, linecolor=COLOR_BASELINE),
        yaxis=dict(tickmode="array", tickvals=ticks, ticktext=[format_currency_abbrev(t) for t in ticks],
                    gridcolor=COLOR_GRIDLINE, zeroline=False),
        hovermode="x unified",
    )
    return fig


def build_category_chart(current_sales: pd.DataFrame) -> go.Figure:
    revenue_by_cat = bm.calculate_revenue_by_dimension(current_sales, "product_category_name").head(10)
    cats = list(revenue_by_cat.index)[::-1]
    values = list(revenue_by_cat.values)[::-1]
    labels = [c.replace("_", " ").title() for c in cats]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=values, colorscale=SEQ_BLUE_SCALE, showscale=False),
        text=[format_currency_abbrev(v) for v in values],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>Revenue: $%{x:,.0f}<extra></extra>",
    ))
    ticks = nice_ticks(max(values, default=0))
    fig.update_layout(**base_layout("Top 10 Categories by Revenue"))
    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=ticks, ticktext=[format_currency_abbrev(t) for t in ticks],
                    gridcolor=COLOR_GRIDLINE, zeroline=False),
        yaxis=dict(showgrid=False),
        showlegend=False,
    )
    return fig


def build_choropleth_chart(current_sales: pd.DataFrame) -> go.Figure:
    revenue_by_state = bm.calculate_revenue_by_dimension(current_sales, "customer_state").reset_index()
    revenue_by_state.columns = ["customer_state", "revenue"]

    fig = go.Figure(go.Choropleth(
        locations=revenue_by_state["customer_state"],
        z=revenue_by_state["revenue"],
        locationmode="USA-states",
        colorscale=SEQ_BLUE_SCALE,
        marker_line_color="white",
        marker_line_width=0.5,
        colorbar=dict(title="Revenue", tickprefix="$", len=0.85),
        hovertemplate="%{location}<br>Revenue: $%{z:,.0f}<extra></extra>",
    ))
    fig.update_layout(**base_layout("Revenue by State"))
    fig.update_layout(
        geo=dict(scope="usa", bgcolor=COLOR_SURFACE, lakecolor=COLOR_SURFACE),
    )
    return fig


def build_satisfaction_chart(order_level: pd.DataFrame) -> go.Figure:
    bucketed = bm.categorize_delivery_speed(
        order_level, bin_edges=DELIVERY_SPEED_BINS, bin_labels=DELIVERY_SPEED_LABELS
    )
    by_bucket = bm.calculate_average_review_score_by_group(bucketed, "delivery_speed_bucket")

    fig = go.Figure(go.Bar(
        x=by_bucket["delivery_speed_bucket"].astype(str),
        y=by_bucket["review_score"],
        marker=dict(color=COLOR_ACCENT),
        text=[f"{v:.2f}" for v in by_bucket["review_score"]],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{x}<br>Avg. review score: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(**base_layout("Satisfaction vs. Delivery Time"))
    fig.update_layout(
        xaxis=dict(title="Delivery time", showgrid=False),
        yaxis=dict(title="Avg. review score (1-5)", range=[0, 5.4], gridcolor=COLOR_GRIDLINE, zeroline=False),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
.stApp {{
    background-color: {COLOR_PAGE};
}}
div[data-testid="stAppViewContainer"] .block-container {{
    padding-top: 3.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}}
.kpi-card, .bottom-card {{
    background-color: {COLOR_SURFACE};
    border: 1px solid rgba(11,11,11,0.10);
    border-radius: 12px;
    padding: 18px 22px;
    box-shadow: 0 1px 2px rgba(11,11,11,0.04);
    display: flex;
    flex-direction: column;
    justify-content: center;
}}
.kpi-card {{ height: 150px; }}
.bottom-card {{ height: 190px; }}
.kpi-title, .bottom-title {{
    font-size: 0.8rem;
    font-weight: 600;
    color: {COLOR_SECONDARY_INK};
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 8px;
}}
.kpi-value {{
    font-size: 1.9rem;
    font-weight: 700;
    color: {COLOR_PRIMARY_INK};
    line-height: 1.15;
}}
.kpi-delta {{
    font-size: 0.95rem;
    font-weight: 600;
    margin-top: 6px;
}}
.kpi-subtitle, .bottom-subtitle {{
    font-size: 0.78rem;
    color: {COLOR_MUTED};
    margin-top: 4px;
}}
.bottom-value {{
    font-size: 2.4rem;
    font-weight: 700;
    color: {COLOR_PRIMARY_INK};
    line-height: 1.1;
}}
.bottom-stars {{
    font-size: 1.7rem;
    letter-spacing: 3px;
    margin-top: 6px;
}}
.section-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {COLOR_PRIMARY_INK};
    margin: 30px 0 14px 0;
}}
.dashboard-title {{
    font-size: 1.9rem;
    font-weight: 700;
    color: {COLOR_PRIMARY_INK};
    margin: 0;
}}
.dashboard-subtitle {{
    font-size: 0.85rem;
    color: {COLOR_MUTED};
    margin-top: 2px;
}}
div[data-testid="stDateInputField"] {{
    margin-top: 6px;
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data + header
# ---------------------------------------------------------------------------
sales_all, reviews_all = load_data()
data_min_date = sales_all["order_purchase_timestamp"].min().date()
data_max_date = sales_all["order_purchase_timestamp"].max().date()

default_end = min(date(2023, 12, 31), data_max_date)
default_start = max(date(2023, 1, 1), data_min_date)

head_col1, head_col2 = st.columns([2.3, 1.2])
with head_col1:
    st.markdown(
        '<div class="dashboard-title">E-Commerce Sales Dashboard</div>'
        '<div class="dashboard-subtitle">Revenue, orders, and customer experience</div>',
        unsafe_allow_html=True,
    )
with head_col2:
    date_range = st.date_input(
        "Date range",
        value=(default_start, default_end),
        min_value=data_min_date,
        max_value=data_max_date,
        label_visibility="collapsed",
    )

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = default_start, default_end

prev_start, prev_end = get_previous_period(start_date, end_date)
prev_label = f"vs {prev_start:%b %d, %Y} – {prev_end:%b %d, %Y}"

current = compute_period_data(sales_all, reviews_all, start_date, end_date)
previous = compute_period_data(sales_all, reviews_all, prev_start, prev_end)

revenue_trend = bm.calculate_percent_change(current["revenue"], previous["revenue"])
aov_trend = bm.calculate_percent_change(current["aov"], previous["aov"])
order_trend = bm.calculate_percent_change(current["order_count"], previous["order_count"])
delivery_trend = bm.calculate_percent_change(current["avg_delivery_days"], previous["avg_delivery_days"])

monthly_rev = monthly_revenue_series(current["sales"])
mom_growth = bm.calculate_month_over_month_growth(monthly_rev)
avg_mom_growth = mom_growth.mean() if mom_growth.notna().any() else float("nan")

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        kpi_card("Total Revenue", format_currency_full(current["revenue"]), revenue_trend, "up", prev_label),
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        kpi_growth_card("Monthly Growth", avg_mom_growth, "Avg. month-over-month, within period"),
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        kpi_card("Average Order Value", format_currency_full(current["aov"]), aov_trend, "up", prev_label),
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        kpi_card("Total Orders", f"{current['order_count']:,}", order_trend, "up", prev_label),
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Charts grid (2x2)
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Performance Overview</div>', unsafe_allow_html=True)
chart_config = {"displayModeBar": False}

row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    with st.container(border=True, height=CHART_CARD_HEIGHT):
        st.plotly_chart(build_revenue_trend_chart(current["sales"], previous["sales"]),
                         use_container_width=True, config=chart_config)
with row1_col2:
    with st.container(border=True, height=CHART_CARD_HEIGHT):
        st.plotly_chart(build_category_chart(current["sales"]),
                         use_container_width=True, config=chart_config)

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    with st.container(border=True, height=CHART_CARD_HEIGHT):
        st.plotly_chart(build_choropleth_chart(current["sales"]),
                         use_container_width=True, config=chart_config)
with row2_col2:
    with st.container(border=True, height=CHART_CARD_HEIGHT):
        st.plotly_chart(build_satisfaction_chart(current["order_level"]),
                         use_container_width=True, config=chart_config)

# ---------------------------------------------------------------------------
# Bottom row
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Delivery & Satisfaction</div>', unsafe_allow_html=True)
b1, b2 = st.columns(2)
with b1:
    st.markdown(
        bottom_metric_card(
            "Average Delivery Time",
            f"{current['avg_delivery_days']:.2f} days" if pd.notna(current["avg_delivery_days"]) else "N/A",
            delivery_trend, "down", prev_label,
        ),
        unsafe_allow_html=True,
    )
with b2:
    st.markdown(review_score_card(current["avg_review_score"]), unsafe_allow_html=True)
