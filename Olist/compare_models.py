"""
Run multiple river models on the same Olist stream and compare them.

Models (six, in roughly increasing order of capability):
  baseline_no_change      - predicts the last-seen label (river dummy)
  baseline_prior          - predicts by class priors (river dummy)
  naive_bayes             - GaussianNB on numerics + OHE cats
  logistic_ovr            - OneVsRest(LogisticRegression), linear baseline
  hoeffding_tree          - non-adaptive Hoeffding Tree
  hoeffding_adaptive      - drift-aware Hoeffding Tree (HAT)
  adaptive_forest         - ARF ensemble (slide 5's "ensemble of HATs")

Design: one pass through the stream. The feature extractor runs ONCE per
event; every model sees the exact same enriched input and is updated in
lockstep. Otherwise we'd be comparing models on different inputs. Per-model
metrics and drift detectors are independent.

Usage:
    python compare_models.py --input ./olist_stream.parquet \
        --reviews ./olist_stream_reviews.parquet \
        --history-csv ./compare_history.csv \
        --summary-csv ./compare_summary.csv
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from river import (
    compose, dummy, forest, linear_model, multiclass,
    naive_bayes, preprocessing, tree,
)

from stream_pipeline import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    StreamFeatureExtractor, TopKAccuracy, DriftMonitor,
)
from river import metrics as rmetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _preprocess():
    """Shared front-end: scale numerics, one-hot the low-card categoricals."""
    return compose.TransformerUnion(
        compose.Select(*NUMERIC_FEATURES) | preprocessing.StandardScaler(),
        compose.Select(*CATEGORICAL_FEATURES) | preprocessing.OneHotEncoder(),
    )


def build_models(arf_n_models: int = 5) -> dict:
    """Factory dict: name -> () -> fresh pipeline.

    Functions (not instances) so each call creates a clean pipeline; useful
    when comparing across different runs or seeds.

    Note: NoChange and Prior don't need preprocessing, but we wrap them
    anyway for a uniform call signature. Cost is negligible.
    """
    return {
        "baseline_no_change": lambda: _preprocess() | dummy.NoChangeClassifier(),
        "baseline_prior":     lambda: _preprocess() | dummy.PriorClassifier(),
        "naive_bayes":        lambda: _preprocess() | naive_bayes.GaussianNB(),
        "logistic_ovr":       lambda: _preprocess() | multiclass.OneVsRestClassifier(
            linear_model.LogisticRegression()
        ),
        "hoeffding_tree":     lambda: _preprocess() | tree.HoeffdingTreeClassifier(
            grace_period=200,
        ),
        "hoeffding_adaptive": lambda: _preprocess() | tree.HoeffdingAdaptiveTreeClassifier(
            grace_period=200, seed=42,
        ),
        "adaptive_forest":    lambda: _preprocess() | forest.ARFClassifier(
            n_models=arf_n_models, seed=42,
        ),
    }


# ---------------------------------------------------------------------------
# Per-model state container
# ---------------------------------------------------------------------------

class ModelState:
    """Bundles a model's pipeline with its metrics and drift monitor."""

    def __init__(self, name: str, pipeline, track_drift: bool = True):
        self.name = name
        self.pipeline = pipeline
        self.accuracy = rmetrics.Accuracy()
        self.macro_f1 = rmetrics.MacroF1()
        self.top5     = TopKAccuracy(k=5)
        self.drift    = DriftMonitor() if track_drift else None
        # Wallclock time spent on predict_one + learn_one — useful for
        # comparing the cost of the strong learners vs baselines.
        self.elapsed_s = 0.0

    def step(self, x, y, n: int, t_sec: float) -> None:
        t0 = time.perf_counter()
        y_pred  = self.pipeline.predict_one(x)
        y_proba = self.pipeline.predict_proba_one(x)
        if y_pred is not None:
            self.accuracy.update(y, y_pred)
            self.macro_f1.update(y, y_pred)
            if self.drift is not None:
                self.drift.update(int(y_pred != y), n=n, t_sec=t_sec)
        if y_proba:
            self.top5.update(y, y_proba)
        self.pipeline.learn_one(x, y)
        self.elapsed_s += time.perf_counter() - t0

    def snapshot(self, n: int) -> dict:
        return {
            "n": n,
            "model": self.name,
            "accuracy": self.accuracy.get(),
            "macro_f1": self.macro_f1.get(),
            "top5_accuracy": self.top5.get(),
        }

    def summary(self) -> dict:
        return {
            "model": self.name,
            "accuracy": self.accuracy.get(),
            "macro_f1": self.macro_f1.get(),
            "top5_accuracy": self.top5.get(),
            "drift_events": len(self.drift.events) if self.drift else 0,
            "elapsed_s": self.elapsed_s,
        }


# ---------------------------------------------------------------------------
# Comparison loop
# ---------------------------------------------------------------------------

def run_comparison(
    df: pd.DataFrame,
    model_factories: dict,
    extractor: StreamFeatureExtractor | None = None,
    reviews_df: pd.DataFrame | None = None,
    print_every: int = 5_000,
    track_drift: bool = True,
) -> tuple[list[ModelState], list[dict]]:
    """Run every model in lockstep over the stream.

    Returns:
        states: list of ModelState (final metrics, drift events, time)
        history: list of per-checkpoint snapshots, long format. One row per
                 (model, checkpoint).
    """
    if extractor is None:
        extractor = StreamFeatureExtractor()

    states = [ModelState(name, factory(), track_drift=track_drift)
              for name, factory in model_factories.items()]
    history: list[dict] = []

    # Merge purchases and reviews into one time-sorted iterator (same logic
    # as progressive_validation in stream_pipeline.py, kept consistent so
    # the comparison is faithful to single-model results).
    purchase_events = (
        (pd.Timestamp(t).timestamp(), 0, "purchase", row)
        for t, row in zip(df["order_purchase_timestamp"], df.to_dict("records"))
    )
    if reviews_df is not None and len(reviews_df) > 0:
        review_events = (
            (pd.Timestamp(t).timestamp(), 1, "review", row)
            for t, row in zip(reviews_df["review_creation_date"],
                              reviews_df.to_dict("records"))
        )
        merged = sorted(list(purchase_events) + list(review_events),
                        key=lambda e: (e[0], e[1]))
    else:
        merged = list(purchase_events)

    label_col = "product_category_name"
    n_purchases = 0
    n_reviews = 0
    t_start = time.perf_counter()

    for t_sec, _, kind, x in merged:
        if kind == "review":
            n_reviews += 1
            extractor.update_on_review(
                order_id=x["order_id"], score=x["review_score"], t=t_sec,
            )
            continue

        n_purchases += 1
        y = x[label_col]
        x_enriched = extractor.transform_one(x)

        for state in states:
            state.step(x_enriched, y, n=n_purchases, t_sec=t_sec)

        extractor.update(x, y)

        if n_purchases % print_every == 0:
            elapsed = time.perf_counter() - t_start
            logger.info(
                "n=%-7d  elapsed=%5.1fs  reviews_seen=%d", n_purchases, elapsed, n_reviews,
            )
            for state in states:
                logger.info(
                    "  %-22s  acc=%.4f  macroF1=%.4f  top5=%.4f  cost=%5.1fs",
                    state.name, state.accuracy.get(), state.macro_f1.get(),
                    state.top5.get(), state.elapsed_s,
                )
                history.append(state.snapshot(n_purchases))

    return states, history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=5_000)
    parser.add_argument("--history-csv", type=Path, default=None,
                        help="Long-format history (n, model, metric...).")
    parser.add_argument("--summary-csv", type=Path, default=None,
                        help="One row per model with final metrics + cost.")
    parser.add_argument("--drift-csv", type=Path, default=None,
                        help="One row per drift firing across all models.")
    parser.add_argument("--arf-n-models", type=int, default=5,
                        help="Number of trees in the ARF ensemble (default 5; "
                             "10 is more accurate but ~2x slower).")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated subset of model names to run. "
                             "Default: all. Useful for quick iteration.")
    parser.add_argument("--no-drift", action="store_true",
                        help="Skip drift detection to save time.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.input.suffix == ".parquet":
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input, parse_dates=["order_purchase_timestamp"])
    if args.limit:
        df = df.head(args.limit)

    reviews_df = None
    if args.reviews is not None:
        if args.reviews.suffix == ".parquet":
            reviews_df = pd.read_parquet(args.reviews)
        else:
            reviews_df = pd.read_csv(args.reviews,
                                     parse_dates=["review_creation_date"])
        if args.limit:
            reviews_df = reviews_df[
                reviews_df["review_creation_date"]
                <= df["order_purchase_timestamp"].max()
            ]

    factories = build_models(arf_n_models=args.arf_n_models)
    if args.models:
        wanted = [m.strip() for m in args.models.split(",")]
        unknown = set(wanted) - set(factories)
        if unknown:
            raise SystemExit(f"Unknown models: {sorted(unknown)}. "
                             f"Available: {sorted(factories)}")
        factories = {k: factories[k] for k in wanted}

    logger.info("Running %d model(s) on %d events: %s",
                len(factories), len(df), list(factories))

    states, history = run_comparison(
        df, factories,
        reviews_df=reviews_df,
        print_every=args.print_every,
        track_drift=not args.no_drift,
    )

    # ---- Write outputs ----
    if args.history_csv:
        pd.DataFrame(history).to_csv(args.history_csv, index=False)
        logger.info("Wrote history to %s", args.history_csv)

    summary = pd.DataFrame([s.summary() for s in states])
    if args.summary_csv:
        summary.to_csv(args.summary_csv, index=False)
        logger.info("Wrote summary to %s", args.summary_csv)

    if args.drift_csv:
        rows = []
        for s in states:
            if s.drift is None:
                continue
            df_d = s.drift.to_dataframe()
            df_d["model"] = s.name
            rows.append(df_d)
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(args.drift_csv, index=False)
            logger.info("Wrote drift events to %s", args.drift_csv)

    # ---- Final summary to stdout ----
    logger.info("=" * 70)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 70)
    summary_sorted = summary.sort_values("top5_accuracy", ascending=False)
    for _, r in summary_sorted.iterrows():
        logger.info(
            "  %-22s  acc=%.4f  macroF1=%.4f  top5=%.4f  drift=%-3d  time=%5.1fs",
            r["model"], r["accuracy"], r["macro_f1"], r["top5_accuracy"],
            int(r["drift_events"]), r["elapsed_s"],
        )


if __name__ == "__main__":
    main()