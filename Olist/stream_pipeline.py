"""
Stream the prepared Olist data through a river pipeline with stateful feature
extraction, and run progressive validation.

The feature extractor is intentionally OUTSIDE the river pipeline so we have
explicit control of the order: read state -> predict -> learn -> update state.

Usage:
    python stream_pipeline.py --input ./olist_stream.parquet
    python stream_pipeline.py --input ./olist_stream.parquet --limit 20000 --print-every 2000
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from river import compose, ensemble, metrics, naive_bayes, preprocessing, tree

logger = logging.getLogger(__name__)


class TopKAccuracy:
    """Streaming top-k accuracy computed from predict_proba_one() output.

    river 0.24 doesn't ship a top-k metric, so we roll our own. For each event:
    take the k labels with the highest predicted probability and score 1 if the
    true label is among them, 0 otherwise.
    """

    def __init__(self, k: int = 5):
        self.k = k
        self._correct = 0
        self._total = 0

    def update(self, y_true, y_proba: dict) -> None:
        if not y_proba:
            return
        top = sorted(y_proba.items(), key=lambda kv: kv[1], reverse=True)[: self.k]
        if any(lbl == y_true for lbl, _ in top):
            self._correct += 1
        self._total += 1

    def get(self) -> float:
        return self._correct / self._total if self._total else 0.0


class DriftMonitor:
    """Run multiple drift detectors over a stream of error indicators.

    The monitor is PASSIVE: it logs drift events but doesn't reset or
    re-train the model. The model has its own drift adaptation
    (HoeffdingAdaptiveTree's internal change detection, ARF's per-tree
    detectors); this monitor gives us an external, comparable view of
    *when* and *which* detectors fired.

    Signal convention: feed `error` where 1 means the prediction was wrong
    and 0 means it was correct. An INCREASE in the mean signals drift.
    """

    def __init__(self, detectors: dict | None = None):
        if detectors is None:
            from river import drift
            detectors = {
                "ADWIN":       drift.ADWIN(),
                "DDM":         drift.binary.DDM(),  # binary in river >=0.20
                "PageHinkley": drift.PageHinkley(),
            }
        self.detectors = detectors
        self.events: list[dict] = []

    def update(self, error: float, n: int, t_sec: float) -> list[str]:
        """Feed one observation. Returns names of detectors that signaled
        drift on this step (usually empty)."""
        fired = []
        for name, det in self.detectors.items():
            det.update(error)
            if det.drift_detected:
                self.events.append({
                    "detector": name,
                    "n": n,
                    "timestamp": pd.Timestamp(t_sec, unit="s"),
                })
                fired.append(name)
        return fired

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.events) if self.events else pd.DataFrame(
            columns=["detector", "n", "timestamp"]
        )

    def save(self, path: Path) -> None:
        self.to_dataframe().to_csv(path, index=False)

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
# These lists must match the keys produced by StreamFeatureExtractor.transform_one
# AND the columns in the prepared parquet. Anything not listed is dropped before
# the model sees it.

NUMERIC_FEATURES = [
    # Calendar (from prep script)
    "hour", "dow", "month", "is_weekend", "days_since_start",
    # Item / order numerics (from prep script)
    "price", "freight_value",
    "payment_value_total", "payment_installments", "payment_methods_count",
    "product_weight_g", "product_length_cm", "product_height_cm",
    "product_width_cm", "product_photos_qty",
    "product_name_lenght", "product_description_lenght",
    # Added by StreamFeatureExtractor (history)
    "customer_order_count", "customer_avg_price",
    "days_since_last_order",
    "item_popularity_7d", "seller_popularity_30d",
    # Added by StreamFeatureExtractor (reviews)
    "customer_review_count", "customer_avg_review_score",
    "product_recent_review_score", "seller_recent_review_score",
]

CATEGORICAL_FEATURES = [
    # From prep script
    "customer_state", "payment_type",
    # Added by StreamFeatureExtractor
    "customer_last_category",
]


# ---------------------------------------------------------------------------
# Stateful feature extractor
# ---------------------------------------------------------------------------

class DecayingCounter:
    """Exponentially-decayed counter keyed by id.

    `add(k, t)` records an event for key k at time t (seconds since epoch).
    `get(k, t)` returns the decayed count at time t. Half-life is in days.
    """

    def __init__(self, half_life_days: float):
        # Decay rate per second such that value halves every `half_life_days` days.
        self.lam = math.log(2.0) / (half_life_days * 86400.0)
        self._value: dict[Any, float] = {}
        self._last_t: dict[Any, float] = {}

    def get(self, key: Any, t: float) -> float:
        if key not in self._value:
            return 0.0
        dt = t - self._last_t[key]
        return self._value[key] * math.exp(-self.lam * dt)

    def add(self, key: Any, t: float, w: float = 1.0) -> None:
        if key in self._value:
            dt = t - self._last_t[key]
            self._value[key] = self._value[key] * math.exp(-self.lam * dt) + w
        else:
            self._value[key] = w
        self._last_t[key] = t

    def __len__(self) -> int:
        return len(self._value)


class DecayingMean:
    """Exponentially-decayed running mean.

    Stores a numerator (sum of values) and denominator (sum of weights),
    both decayed by exp(-lambda * dt) at every update. Querying without an
    update doesn't mutate state — the most-recent ratio is returned as-is,
    which is the right behavior for "the most recent signal we have."
    """

    def __init__(self, half_life_days: float):
        self.lam = math.log(2.0) / (half_life_days * 86400.0)
        self._num: dict[Any, float] = {}
        self._den: dict[Any, float] = {}
        self._last_t: dict[Any, float] = {}

    def get(self, key: Any, default: float | None = None) -> float | None:
        if key not in self._num or self._den[key] <= 0:
            return default
        return self._num[key] / self._den[key]

    def add(self, key: Any, value: float, t: float, w: float = 1.0) -> None:
        if key in self._num:
            dt = t - self._last_t[key]
            decay = math.exp(-self.lam * dt)
            self._num[key] = self._num[key] * decay + value * w
            self._den[key] = self._den[key] * decay + w
        else:
            self._num[key] = value * w
            self._den[key] = w
        self._last_t[key] = t

    def __len__(self) -> int:
        return len(self._num)


@dataclass
class _CustState:
    count: int = 0
    sum_price: float = 0.0
    last_category: str = "NEW"
    last_t: float = 0.0


class CustomerHistory:
    """Per-customer running stats. Reads return state BEFORE the current event."""

    def __init__(self):
        self._state: dict[str, _CustState] = {}

    def features(self, cust_id: str, t: float) -> dict:
        s = self._state.get(cust_id)
        if s is None:
            return {
                "customer_order_count": 0,
                "customer_avg_price": 0.0,
                "customer_last_category": "NEW",
                "days_since_last_order": -1.0,  # sentinel for "never"
            }
        return {
            "customer_order_count": s.count,
            "customer_avg_price": s.sum_price / s.count,
            "customer_last_category": s.last_category,
            "days_since_last_order": (t - s.last_t) / 86400.0,
        }

    def update(self, cust_id: str, price: float, category: str, t: float) -> None:
        s = self._state.get(cust_id)
        if s is None:
            self._state[cust_id] = _CustState(
                count=1, sum_price=float(price),
                last_category=category, last_t=t,
            )
        else:
            s.count += 1
            s.sum_price += float(price)
            s.last_category = category
            s.last_t = t

    def __len__(self) -> int:
        return len(self._state)


class OrderRegistry:
    """Maps order_id -> list of (customer_unique_id, product_id, seller_id).

    Populated as purchase events stream by. Consulted when a review event
    arrives so we can credit the right customer/product/seller. One order
    can contain several items, so each lookup returns a list.
    """

    def __init__(self):
        self._items: dict[str, list[tuple[str, str, str]]] = {}

    def register(self, order_id: str, customer_id: str,
                 product_id: str, seller_id: str) -> None:
        self._items.setdefault(order_id, []).append((customer_id, product_id, seller_id))

    def get(self, order_id: str) -> list[tuple[str, str, str]]:
        return self._items.get(order_id, [])

    def __len__(self) -> int:
        return len(self._items)


class ReviewState:
    """Per-customer / per-product / per-seller review aggregates.

    Customer aggregates use no decay — review tendency is treated as a
    personality trait. Product and seller aggregates decay because reputation
    moves; sellers move slower than products, so the half-life is longer.
    """

    def __init__(self,
                 product_half_life_days: float = 30.0,
                 seller_half_life_days: float = 60.0):
        self._cust_count: dict[str, int] = {}
        self._cust_sum: dict[str, float] = {}
        self.product = DecayingMean(product_half_life_days)
        self.seller  = DecayingMean(seller_half_life_days)

    def features(self, customer_id: str, product_id: str, seller_id: str) -> dict:
        n = self._cust_count.get(customer_id, 0)
        return {
            "customer_review_count": n,
            "customer_avg_review_score": (
                self._cust_sum[customer_id] / n if n > 0 else -1.0
            ),
            "product_recent_review_score": self.product.get(product_id, default=-1.0),
            "seller_recent_review_score":  self.seller.get(seller_id,  default=-1.0),
        }

    def update_customer(self, customer_id: str, score: float) -> None:
        self._cust_count[customer_id] = self._cust_count.get(customer_id, 0) + 1
        self._cust_sum[customer_id]   = self._cust_sum.get(customer_id, 0.0) + float(score)

    def update_product(self, product_id: str, score: float, t: float) -> None:
        self.product.add(product_id, float(score), t)

    def update_seller(self, seller_id: str, score: float, t: float) -> None:
        self.seller.add(seller_id, float(score), t)

    def __len__(self) -> int:
        return len(self._cust_count)


@dataclass
class StreamFeatureExtractor:
    """Adds stateful features to a raw event dict.

    Use:
        x_enriched = extractor.transform_one(x)   # before predict
        ...predict and learn on x_enriched...
        extractor.update(x, y)                    # after learn
        # ...and on review events arriving in the stream:
        extractor.update_on_review(order_id, score, t)
    """
    item_half_life_days: float = 7.0
    seller_half_life_days: float = 30.0
    product_review_half_life_days: float = 30.0
    seller_review_half_life_days: float = 60.0

    customers: CustomerHistory = field(default_factory=CustomerHistory)
    item_pop: DecayingCounter = field(init=False)
    seller_pop: DecayingCounter = field(init=False)
    reviews: ReviewState = field(init=False)
    orders: OrderRegistry = field(default_factory=OrderRegistry)

    def __post_init__(self):
        self.item_pop = DecayingCounter(self.item_half_life_days)
        self.seller_pop = DecayingCounter(self.seller_half_life_days)
        self.reviews = ReviewState(
            product_half_life_days=self.product_review_half_life_days,
            seller_half_life_days=self.seller_review_half_life_days,
        )

    @staticmethod
    def _to_seconds(ts) -> float:
        # Handles both pandas.Timestamp (from parquet) and str (from CSV).
        return pd.Timestamp(ts).timestamp()

    def transform_one(self, x: dict) -> dict:
        t = self._to_seconds(x["order_purchase_timestamp"])
        out = dict(x)
        out.update(self.customers.features(x["customer_unique_id"], t))
        out["item_popularity_7d"] = self.item_pop.get(x["product_id"], t)
        out["seller_popularity_30d"] = self.seller_pop.get(x["seller_id"], t)
        out.update(self.reviews.features(
            x["customer_unique_id"], x["product_id"], x["seller_id"],
        ))
        return out

    def update(self, x: dict, y: str) -> None:
        t = self._to_seconds(x["order_purchase_timestamp"])
        self.customers.update(
            cust_id=x["customer_unique_id"],
            price=x["price"],
            category=y,
            t=t,
        )
        self.item_pop.add(x["product_id"], t)
        self.seller_pop.add(x["seller_id"], t)
        # Remember which (customer, product, seller) belong to this order so
        # we can credit them when a review for this order arrives later.
        self.orders.register(
            order_id=x["order_id"],
            customer_id=x["customer_unique_id"],
            product_id=x["product_id"],
            seller_id=x["seller_id"],
        )

    def update_on_review(self, order_id: str, score: float, t: float) -> None:
        """Apply a review event. Updates customer / product / seller aggregates
        for every item in the reviewed order. Reviews for orders we never saw
        (e.g., before the stream started) are silently ignored."""
        for cust_id, prod_id, seller_id in self.orders.get(order_id):
            self.reviews.update_customer(cust_id, score)
            self.reviews.update_product(prod_id, score, t)
            self.reviews.update_seller(seller_id, score, t)


# ---------------------------------------------------------------------------
# River pipeline
# ---------------------------------------------------------------------------

def build_pipeline(model=None):
    """Numerics through StandardScaler, low-card cats through OneHotEncoder,
    everything else dropped. The classifier defaults to a Hoeffding Adaptive
    Tree, which is drift-aware out of the box.
    """
    if model is None:
        model = tree.HoeffdingAdaptiveTreeClassifier(grace_period=200, seed=42)

    return compose.Pipeline(
        compose.TransformerUnion(
            compose.Select(*NUMERIC_FEATURES) | preprocessing.StandardScaler(),
            compose.Select(*CATEGORICAL_FEATURES) | preprocessing.OneHotEncoder(),
        ),
        model,
    )


def build_baselines() -> dict:
    """Return a dict of name -> river pipeline for the comparison plot."""
    return {
        "naive_bayes": compose.Pipeline(
            compose.TransformerUnion(
                compose.Select(*NUMERIC_FEATURES) | preprocessing.StandardScaler(),
                compose.Select(*CATEGORICAL_FEATURES) | preprocessing.OneHotEncoder(),
            ),
            naive_bayes.GaussianNB(),
        ),
        "hoeffding_adaptive_tree": build_pipeline(),
        "adaptive_random_forest": build_pipeline(
            ensemble.AdaptiveRandomForestClassifier(n_models=10, seed=42)
        ),
    }


# ---------------------------------------------------------------------------
# Progressive validation
# ---------------------------------------------------------------------------

def progressive_validation(
    df: pd.DataFrame,
    model_pipeline,
    extractor: StreamFeatureExtractor,
    reviews_df: pd.DataFrame | None = None,
    drift_monitor: DriftMonitor | None = None,
    print_every: int = 5_000,
    history_path: Path | None = None,
) -> dict:
    """Predict-then-learn loop. If `reviews_df` is given, review events are
    interleaved with purchase events by timestamp and used to update review
    state (no prediction is made on review events). If `drift_monitor` is
    given, every per-event error signal is fed to its detectors.

    Returns a dict of final metric values and writes a per-step CSV if
    `history_path` is provided.
    """

    accuracy = metrics.Accuracy()
    macro_f1 = metrics.MacroF1()
    top5     = TopKAccuracy(k=5)

    history_rows = []

    label_col = "product_category_name"

    # Build a single time-sorted iterator over both event types. Reviews are
    # given a tiny extra tiebreak so that if a review and a purchase share a
    # timestamp, the purchase is processed first (a review created at the same
    # second as a purchase is, by convention, AFTER it).
    purchase_events = (
        (pd.Timestamp(t).timestamp(), 0, "purchase", row)
        for t, row in zip(df["order_purchase_timestamp"], df.to_dict("records"))
    )
    if reviews_df is not None and len(reviews_df) > 0:
        review_events = (
            (pd.Timestamp(t).timestamp(), 1, "review", row)
            for t, row in zip(
                reviews_df["review_creation_date"], reviews_df.to_dict("records"),
            )
        )
        merged = sorted(  # merge two pre-sorted sequences
            list(purchase_events) + list(review_events),
            key=lambda e: (e[0], e[1]),
        )
    else:
        merged = list(purchase_events)

    n_purchases = 0
    n_reviews = 0
    n_orphan_reviews = 0  # reviews for orders we never saw

    for t_sec, _, kind, x in merged:

        if kind == "review":
            n_reviews += 1
            credited = extractor.orders.get(x["order_id"])
            if not credited:
                n_orphan_reviews += 1
            extractor.update_on_review(
                order_id=x["order_id"],
                score=x["review_score"],
                t=t_sec,
            )
            continue

        # Purchase event
        n_purchases += 1
        y = x[label_col]

        # 1) Read state -> enrich
        x_enriched = extractor.transform_one(x)

        # 2) Predict (skip if model has seen no data yet, to avoid early NaNs)
        y_pred = model_pipeline.predict_one(x_enriched)
        y_proba = model_pipeline.predict_proba_one(x_enriched)

        if y_pred is not None:
            accuracy.update(y, y_pred)
            macro_f1.update(y, y_pred)
            if drift_monitor is not None:
                drift_monitor.update(
                    error=int(y_pred != y),
                    n=n_purchases,
                    t_sec=t_sec,
                )
        if y_proba:
            top5.update(y, y_proba)

        # 3) Learn
        model_pipeline.learn_one(x_enriched, y)

        # 4) Update extractor state with the just-observed event
        extractor.update(x, y)

        if n_purchases % print_every == 0:
            logger.info(
                "n=%-7d  acc=%.4f  macroF1=%.4f  top5=%.4f  "
                "customers=%d  items=%d  sellers=%d  reviews_seen=%d",
                n_purchases, accuracy.get(), macro_f1.get(), top5.get(),
                len(extractor.customers), len(extractor.item_pop),
                len(extractor.seller_pop), n_reviews,
            )
            history_rows.append({
                "n": n_purchases,
                "accuracy": accuracy.get(),
                "macro_f1": macro_f1.get(),
                "top5_accuracy": top5.get(),
                "reviews_seen": n_reviews,
            })

    if history_path is not None:
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        logger.info("Wrote running metrics to %s", history_path)

    if n_orphan_reviews:
        logger.info(
            "%d/%d review events were for orders never observed (skipped).",
            n_orphan_reviews, n_reviews,
        )

    return {
        "n_purchases": n_purchases,
        "n_reviews": n_reviews,
        "accuracy": accuracy.get(),
        "macro_f1": macro_f1.get(),
        "top5_accuracy": top5.get(),
        "drift_events": len(drift_monitor.events) if drift_monitor else 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to the parquet/csv produced by prepare_olist.py")
    parser.add_argument("--reviews", type=Path, default=None,
                        help="Optional reviews stream (sibling *_reviews.parquet "
                             "from prepare_olist.py). If omitted, the pipeline "
                             "runs without review features.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N events (debugging).")
    parser.add_argument("--print-every", type=int, default=5_000)
    parser.add_argument("--history-csv", type=Path, default=None,
                        help="Optional path to write per-step running metrics.")
    parser.add_argument("--drift-events-csv", type=Path, default=None,
                        help="If set, run drift detectors (ADWIN, DDM, PageHinkley) "
                             "on the per-event error stream and write firings here.")
    parser.add_argument("--item-half-life-days", type=float, default=7.0)
    parser.add_argument("--seller-half-life-days", type=float, default=30.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.input.suffix == ".parquet":
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input, parse_dates=["order_purchase_timestamp"])

    if args.limit:
        df = df.head(args.limit)
    logger.info("Loaded %d events. Time span %s -> %s",
                len(df),
                df["order_purchase_timestamp"].min(),
                df["order_purchase_timestamp"].max())

    reviews_df = None
    if args.reviews is not None:
        if args.reviews.suffix == ".parquet":
            reviews_df = pd.read_parquet(args.reviews)
        else:
            reviews_df = pd.read_csv(args.reviews, parse_dates=["review_creation_date"])
        # Trim reviews to the same time horizon as the (possibly limited) purchase frame.
        if args.limit:
            reviews_df = reviews_df[
                reviews_df["review_creation_date"]
                <= df["order_purchase_timestamp"].max()
            ]
        logger.info("Loaded %d review events.", len(reviews_df))

    extractor = StreamFeatureExtractor(
        item_half_life_days=args.item_half_life_days,
        seller_half_life_days=args.seller_half_life_days,
    )
    pipeline = build_pipeline()
    drift_monitor = DriftMonitor() if args.drift_events_csv else None

    final = progressive_validation(
        df, pipeline, extractor,
        reviews_df=reviews_df,
        drift_monitor=drift_monitor,
        print_every=args.print_every,
        history_path=args.history_csv,
    )
    if drift_monitor is not None:
        drift_monitor.save(args.drift_events_csv)
        logger.info(
            "Drift monitor: %d total firings across %d detectors -> %s",
            len(drift_monitor.events),
            len(drift_monitor.detectors),
            args.drift_events_csv,
        )
    logger.info("FINAL: %s", final)


if __name__ == "__main__":
    main()