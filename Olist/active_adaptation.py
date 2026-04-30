"""
Active drift adaptation: wrap any river classifier with warning/drift-triggered
model swapping.

Strategy: when the detector signals WARNING, spin up a shadow model that learns
in parallel. When the detector signals DRIFT, compare main vs shadow on recent
performance and promote whichever is better. If no shadow is ready (e.g.,
detector has only one stage like ADWIN), fall back to resetting the main model.

This is the strategy from Lu et al. ("Learning under Concept Drift: A Review",
slide 9 of the project deck): a background model trained during the warning
window, evaluated, and swapped in.
"""

from __future__ import annotations

import collections
from typing import Callable, Any


class ActiveAdaptiveClassifier:
    """Active drift adaptation wrapper.

    Args:
        model_factory: zero-arg callable returning a fresh river classifier
            (or pipeline). Called to create the initial model and any
            replacements.
        detector_factory: zero-arg callable returning a fresh drift detector.
            Detector should accept ``update(error)`` where error is 0 or 1.
            If the detector exposes ``warning_detected``, the full
            warning→shadow→drift workflow is used; otherwise we fall back to
            single-stage reset-on-drift.
        recent_window: number of most-recent events used to compare main vs
            shadow accuracy at promotion time. 200 is a reasonable default —
            long enough to be statistically meaningful, short enough that a
            shadow trained during the warning period has comparable data.
        min_shadow_events: minimum events the shadow must have learned from
            before it can be promoted. Prevents promoting a shadow that
            barely saw any data because warning and drift fired in rapid
            succession.

    Implements the river classifier protocol: ``predict_one``,
    ``predict_proba_one``, ``learn_one``.
    """

    def __init__(
        self,
        model_factory: Callable[[], Any],
        detector_factory: Callable[[], Any],
        recent_window: int = 200,
        min_shadow_events: int = 50,
    ):
        self.model_factory = model_factory
        self.detector_factory = detector_factory
        self.recent_window = recent_window
        self.min_shadow_events = min_shadow_events

        self.main = model_factory()
        self.detector = detector_factory()
        self.shadow = None
        self.shadow_n = 0

        # Rolling correctness windows for main and shadow. Comparing all-time
        # accuracy would be wrong — main has been learning for the whole
        # stream, shadow only since the warning fired.
        self.main_recent = collections.deque(maxlen=recent_window)
        self.shadow_recent = collections.deque(maxlen=recent_window)

        # Bookkeeping for diagnostics. Lets the caller see what the wrapper
        # actually did, not just what the detector flagged.
        self.events: list[dict] = []
        self._n = 0

    # --- river classifier API ------------------------------------------------

    def predict_one(self, x):
        return self.main.predict_one(x)

    def predict_proba_one(self, x):
        return self.main.predict_proba_one(x)

    def learn_one(self, x, y):
        self._n += 1

        # Record correctness BEFORE learning so it reflects what the model
        # actually predicted at decision time.
        y_pred_main = self.main.predict_one(x)
        if y_pred_main is not None:
            self.main_recent.append(int(y_pred_main == y))

        if self.shadow is not None:
            y_pred_shadow = self.shadow.predict_one(x)
            if y_pred_shadow is not None:
                self.shadow_recent.append(int(y_pred_shadow == y))
            self.shadow.learn_one(x, y)
            self.shadow_n += 1

        self.main.learn_one(x, y)

        # Feed the error signal to the detector.
        if y_pred_main is not None:
            error = int(y_pred_main != y)
            self.detector.update(error)

            warning = getattr(self.detector, "warning_detected", False)
            drift = self.detector.drift_detected

            if warning and self.shadow is None:
                # Warning fired: spin up a candidate. We don't touch main
                # yet — it's still the deployed model.
                self.shadow = self.model_factory()
                self.shadow_n = 0
                self.shadow_recent.clear()
                self.events.append({
                    "n": self._n, "kind": "warning_shadow_started",
                })

            if drift:
                self._handle_drift()

        return self

    # --- internals -----------------------------------------------------------

    def _handle_drift(self) -> None:
        """Decide what to do when drift fires."""
        promoted = False

        if self.shadow is not None and self.shadow_n >= self.min_shadow_events:
            # We have a shadow with enough data — compare and decide.
            main_acc = (
                sum(self.main_recent) / len(self.main_recent)
                if self.main_recent else 0.0
            )
            shadow_acc = (
                sum(self.shadow_recent) / len(self.shadow_recent)
                if self.shadow_recent else 0.0
            )
            if shadow_acc > main_acc:
                self.main = self.shadow
                # The shadow's recent record becomes main's record going
                # forward — these are the predictions THIS model has made.
                self.main_recent = self.shadow_recent
                promoted = True
                self.events.append({
                    "n": self._n, "kind": "drift_shadow_promoted",
                    "main_recent_acc": main_acc,
                    "shadow_recent_acc": shadow_acc,
                    "shadow_n": self.shadow_n,
                })
            else:
                self.events.append({
                    "n": self._n, "kind": "drift_shadow_rejected",
                    "main_recent_acc": main_acc,
                    "shadow_recent_acc": shadow_acc,
                    "shadow_n": self.shadow_n,
                })
        else:
            # No usable shadow — fall back to a hard reset of main. This
            # branch fires for single-stage detectors (ADWIN, PageHinkley)
            # and for two-stage detectors where warning and drift fired in
            # rapid succession.
            self.main = self.model_factory()
            self.main_recent.clear()
            self.events.append({
                "n": self._n, "kind": "drift_main_reset",
                "shadow_present": self.shadow is not None,
                "shadow_n": self.shadow_n,
            })

        # Either way, drop the shadow and make a fresh detector. A stale
        # detector that just fired would otherwise keep firing on the next
        # few events as it re-stabilizes.
        self.shadow = None
        self.shadow_n = 0
        self.shadow_recent.clear()
        self.detector = self.detector_factory()

        _ = promoted  # explicitly note: branch outcome already in events