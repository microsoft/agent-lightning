# Copyright (c) Microsoft. All rights reserved.

"""Logging callback for GEPA optimization within Agent Lightning."""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class LightningGEPACallback:
    """GEPA callback that logs optimization progress via the standard logger.

    Implements the hooks from GEPA's ``GEPACallback`` protocol, forwarding
    each event to Python's logging system. All methods are intentionally
    defensive—exceptions are caught and logged so that a callback failure
    never aborts the optimization run.

    When W&B experiment tracking is enabled (via ``use_wandb``), this callback
    also logs per-iteration progress metrics (pareto front aggregate, best
    candidate score, number of candidates) so that optimization progress is
    visible across every iteration—not only when a new candidate is accepted.
    """

    def __init__(self, *, use_wandb: bool = False) -> None:
        self._use_wandb = use_wandb

    def on_optimization_start(self, event: Dict[str, Any]) -> None:
        logger.info("GEPA optimization started")

    def on_optimization_end(self, event: Dict[str, Any]) -> None:
        logger.info("GEPA optimization ended")

    def on_iteration_start(self, event: Dict[str, Any]) -> None:
        iteration = event.get("iteration", "?")
        logger.info("GEPA iteration %s started", iteration)

    def on_iteration_end(self, event: Dict[str, Any]) -> None:
        iteration = event.get("iteration", "?")
        logger.info("GEPA iteration %s ended", iteration)
        if self._use_wandb:
            self._log_iteration_metrics(event)

    def on_candidate_selected(self, event: Dict[str, Any]) -> None:
        logger.debug("GEPA candidate selected: %s", event)

    def on_candidate_accepted(self, event: Dict[str, Any]) -> None:
        logger.info("GEPA candidate accepted: %s", event)

    def on_candidate_rejected(self, event: Dict[str, Any]) -> None:
        logger.debug("GEPA candidate rejected: %s", event)

    def on_evaluation_start(self, event: Dict[str, Any]) -> None:
        logger.debug("GEPA evaluation started")

    def on_evaluation_end(self, event: Dict[str, Any]) -> None:
        logger.debug("GEPA evaluation ended")

    def on_evaluation_skipped(self, event: Dict[str, Any]) -> None:
        logger.debug("GEPA evaluation skipped: %s", event)

    def on_valset_evaluated(self, event: Dict[str, Any]) -> None:
        logger.debug("GEPA valset evaluated: %s", event)

    def on_reflective_dataset_built(self, event: Dict[str, Any]) -> None:
        logger.debug("GEPA reflective dataset built")

    def on_proposal_start(self, event: Dict[str, Any]) -> None:
        logger.debug("GEPA proposal started")

    def on_proposal_end(self, event: Dict[str, Any]) -> None:
        logger.debug("GEPA proposal ended")

    def on_merge_attempted(self, event: Dict[str, Any]) -> None:
        logger.debug("GEPA merge attempted")

    def on_merge_accepted(self, event: Dict[str, Any]) -> None:
        logger.info("GEPA merge accepted")

    def on_merge_rejected(self, event: Dict[str, Any]) -> None:
        logger.debug("GEPA merge rejected")

    def on_pareto_front_updated(self, event: Dict[str, Any]) -> None:
        logger.debug("GEPA Pareto front updated: %s", event)

    def on_state_saved(self, event: Dict[str, Any]) -> None:
        logger.debug("GEPA state saved")

    def on_budget_updated(self, event: Dict[str, Any]) -> None:
        remaining = event.get("metric_calls_remaining", "?")
        logger.info("GEPA budget updated, remaining: %s", remaining)

    def on_error(self, event: Dict[str, Any]) -> None:
        error = event.get("exception", "unknown")
        logger.error("GEPA error: %s", error)

    def on_minibatch_sampled(self, event: Dict[str, Any]) -> None:
        logger.debug("GEPA minibatch sampled: %s", event)

    def _log_iteration_metrics(self, event: Dict[str, Any]) -> None:
        """Log per-iteration progress metrics to W&B.

        Extracts running totals from ``GEPAState`` so that metrics like
        pareto front aggregate and best candidate score are logged at every
        iteration, not only when a new candidate is accepted.
        """
        try:
            import wandb  # type: ignore[reportMissingImports]

            if wandb.run is None:
                return

            state = event.get("state")
            if state is None:
                return

            iteration: int = event.get("iteration", 0)

            # Pareto front aggregate (average of per-example best scores)
            pareto_scores = list(state.pareto_front_valset.values())
            pareto_agg = sum(pareto_scores) / len(pareto_scores) if pareto_scores else 0.0

            # Best single-candidate aggregate score on valset
            val_scores = state.program_full_scores_val_set
            best_candidate_score = max(val_scores) if val_scores else 0.0

            metrics: Dict[str, Any] = {
                "agl/pareto_front_agg": pareto_agg,
                "agl/best_candidate_valset_score": best_candidate_score,
                "agl/num_candidates": len(state.program_candidates),
                "agl/total_metric_calls": state.total_num_evals,
                "agl/proposal_accepted": event.get("proposal_accepted", False),
            }

            wandb.log(metrics, step=iteration)
        except Exception as e:
            logger.debug("Failed to log iteration metrics to W&B: %s", e)
