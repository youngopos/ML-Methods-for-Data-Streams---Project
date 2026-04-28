"""
Prepare the Brazilian Olist dataset for streaming purchase prediction with river.

Output: a single chronologically-sorted file where each row is one (order, item)
event with features known at purchase time, plus the product category as the
label. This file IS the stream — feed it row-by-row into river.

Usage:
    python prepare_olist.py --input-dir ./olist --output ./olist_stream.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

# Columns we keep from each file. Anything from order_reviews and any
# delivery/approval timestamp is post-purchase information and would leak the
# label or future state, so we never load those columns at all.
ORDERS_COLS = [
    "order_id", "customer_id", "order_status", "order_purchase_timestamp",
]
ITEMS_COLS = [
    "order_id", "order_item_id", "product_id", "seller_id",
    "price", "freight_value",
]
PRODUCTS_COLS = [
    "product_id", "product_category_name",
    "product_weight_g", "product_length_cm", "product_height_cm",
    "product_width_cm", "product_photos_qty",
    "product_name_lenght", "product_description_lenght",  # sic — typos in source
]
CUSTOMERS_COLS = [
    "customer_id", "customer_unique_id",
    "customer_zip_code_prefix", "customer_state",
]
PAYMENTS_COLS = [
    "order_id", "payment_sequential", "payment_type",
    "payment_installments", "payment_value",
]
# Reviews: only columns known at review-creation time. We deliberately do NOT
# load review_comment_*; those are useful but require text features and aren't
# needed for the score-based aggregates.
REVIEWS_COLS = ["order_id", "review_score", "review_creation_date"]

# 'canceled' and 'unavailable' orders are dropped — they're noise for a
# purchase-prediction target. Adjust here if you want to keep them as negatives.
VALID_STATUSES = {"delivered", "shipped", "invoiced", "processing", "approved"}

logger = logging.getLogger(__name__)


def load(input_dir: Path) -> dict[str, pd.DataFrame]:
    """Load only the columns we need from each CSV."""
    files = {
        "orders":    ("olist_orders_dataset.csv",          ORDERS_COLS),
        "items":     ("olist_order_items_dataset.csv",     ITEMS_COLS),
        "products":  ("olist_products_dataset.csv",        PRODUCTS_COLS),
        "customers": ("olist_customers_dataset.csv",       CUSTOMERS_COLS),
        "payments":  ("olist_order_payments_dataset.csv",  PAYMENTS_COLS),
        "reviews":   ("olist_order_reviews_dataset.csv",   REVIEWS_COLS),
    }
    out = {}
    for key, (fname, cols) in files.items():
        path = input_dir / fname
        logger.info("Loading %s", path.name)
        out[key] = pd.read_csv(path, usecols=cols)
        logger.info("  -> %d rows", len(out[key]))
    return out


def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """An order can have several payment rows (e.g. voucher + card). Collapse
    so each order_id appears exactly once: keep the dominant method by value
    and add total/breakdown aggregates."""
    # Primary payment = the row with the largest payment_value for the order.
    idx = payments.groupby("order_id")["payment_value"].idxmax()
    primary = payments.loc[idx, ["order_id", "payment_type", "payment_installments"]]

    totals = (
        payments.groupby("order_id")
        .agg(
            payment_value_total=("payment_value", "sum"),
            payment_methods_count=("payment_sequential", "nunique"),
        )
        .reset_index()
    )
    return primary.merge(totals, on="order_id", how="inner")


def add_calendar_features(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Calendar features derived only from the timestamp. Safe to compute up
    front because they depend on no other event in the stream."""
    ts = df[ts_col]
    return df.assign(
        hour=ts.dt.hour.astype("int8"),
        dow=ts.dt.dayofweek.astype("int8"),
        month=ts.dt.month.astype("int8"),
        is_weekend=(ts.dt.dayofweek >= 5).astype("int8"),
        days_since_start=(ts - ts.min()).dt.total_seconds() / 86400.0,
    )


def collapse_rare_categories(
    df: pd.DataFrame, col: str, min_count: int, other_label: str = "other"
) -> pd.DataFrame:
    """Merge categories with fewer than `min_count` events into a single bucket.
    Keeps the label space manageable for stream classifiers — without this you
    have ~73 classes, many with single-digit support."""
    counts = df[col].value_counts()
    keep = set(counts[counts >= min_count].index)
    rare_mask = ~df[col].isin(keep)
    logger.info(
        "Collapsing %d rare categories (%d rows) below count=%d into '%s'",
        int((counts < min_count).sum()), int(rare_mask.sum()), min_count, other_label,
    )
    df.loc[rare_mask, col] = other_label
    return df


def build_stream_frame(
    raw: dict[str, pd.DataFrame],
    min_category_count: int = 50,
) -> pd.DataFrame:
    orders, items, products = raw["orders"], raw["items"], raw["products"]
    customers, payments = raw["customers"], raw["payments"]

    # Parse timestamp once.
    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], errors="coerce"
    )

    # Drop orders we shouldn't learn from.
    n0 = len(orders)
    orders = orders[orders["order_status"].isin(VALID_STATUSES)]
    orders = orders.dropna(subset=["order_purchase_timestamp"])
    logger.info("Filtered orders by status & timestamp: %d -> %d", n0, len(orders))

    # Map per-order customer_id to stable customer_unique_id (this is the only
    # way to detect repeat buyers).
    orders = orders.merge(customers, on="customer_id", how="left")

    # Aggregate payments to one row per order.
    pay = aggregate_payments(payments)
    orders = orders.merge(pay, on="order_id", how="left")

    # Each row of order_items is one stream event. An order with 3 items emits
    # 3 events sharing a timestamp.
    df = items.merge(orders, on="order_id", how="inner")
    df = df.merge(products, on="product_id", how="left")

    # Drop rows without a label.
    n0 = len(df)
    df = df.dropna(subset=["product_category_name"])
    logger.info("Dropped %d rows missing product_category_name", n0 - len(df))

    # Calendar features.
    df = add_calendar_features(df, "order_purchase_timestamp")

    # Tame the label space.
    df = collapse_rare_categories(df, "product_category_name", min_category_count)

    # Sort chronologically — this ordering IS the stream. Use a stable sort so
    # items within the same order keep their original order_item_id sequence.
    df = df.sort_values("order_purchase_timestamp", kind="mergesort").reset_index(drop=True)

    feature_cols = [
        # Identifiers — kept for stateful features in the stream layer
        # (per-customer history, item popularity, graph features). NOT meant
        # to be passed as raw inputs to the classifier.
        "order_id", "order_item_id", "customer_unique_id", "product_id", "seller_id",
        # Time
        "order_purchase_timestamp",
        "hour", "dow", "month", "is_weekend", "days_since_start",
        # Numerics
        "price", "freight_value",
        "payment_value_total", "payment_installments", "payment_methods_count",
        "product_weight_g", "product_length_cm", "product_height_cm",
        "product_width_cm", "product_photos_qty",
        "product_name_lenght", "product_description_lenght",
        # Categoricals
        "customer_state", "customer_zip_code_prefix", "payment_type",
        # Label
        "product_category_name",
    ]
    return df[feature_cols]


def build_reviews_frame(reviews: pd.DataFrame) -> pd.DataFrame:
    """The reviews table as its own time-sorted stream.

    Each row is one review event. Stream-time = review_creation_date (NOT the
    timestamp of the order being reviewed — the review only becomes a usable
    feature once it has been written).
    """
    out = reviews.copy()
    out["review_creation_date"] = pd.to_datetime(
        out["review_creation_date"], errors="coerce"
    )
    n0 = len(out)
    out = out.dropna(subset=["review_creation_date", "review_score", "order_id"])
    if n0 != len(out):
        logger.info("Dropped %d reviews with missing fields", n0 - len(out))
    out = out.sort_values("review_creation_date", kind="mergesort").reset_index(drop=True)
    return out[["order_id", "review_score", "review_creation_date"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing the Olist CSV files.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path. Use .parquet (recommended) or .csv")
    parser.add_argument("--min-category-count", type=int, default=50,
                        help="Categories with fewer than this many events become 'other'.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    raw = load(args.input_dir)
    df = build_stream_frame(raw, min_category_count=args.min_category_count)

    logger.info("Final stream frame: %d rows, %d cols", *df.shape)
    logger.info(
        "Time span: %s  ->  %s",
        df["order_purchase_timestamp"].min(),
        df["order_purchase_timestamp"].max(),
    )
    logger.info(
        "Label distribution (top 10):\n%s",
        df["product_category_name"].value_counts().head(10).to_string(),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".parquet":
        df.to_parquet(args.output, index=False)
    elif args.output.suffix == ".csv":
        df.to_csv(args.output, index=False)
    else:
        raise ValueError(f"Unsupported output extension: {args.output.suffix}")

    logger.info("Wrote %s", args.output)

    # Reviews — written to a sibling file with `_reviews` inserted before the
    # extension. Skipped if the input directory has no reviews CSV.
    if "reviews" in raw and not raw["reviews"].empty:
        reviews_df = build_reviews_frame(raw["reviews"])
        reviews_path = args.output.with_name(
            args.output.stem + "_reviews" + args.output.suffix
        )
        if args.output.suffix == ".parquet":
            reviews_df.to_parquet(reviews_path, index=False)
        else:
            reviews_df.to_csv(reviews_path, index=False)
        logger.info(
            "Wrote %d review events spanning %s -> %s to %s",
            len(reviews_df),
            reviews_df["review_creation_date"].min(),
            reviews_df["review_creation_date"].max(),
            reviews_path,
        )


if __name__ == "__main__":
    main()