"""
evaluator.py — one ruler, three lines.

Mean next-token cross-entropy over the SUPERVISED tokens of a frozen val set. The
same class serves all three lines, because once the data source has written
loss_weights there is nothing modality-specific left to measure: MixedDataLoader
has already turned weight-0 positions into IGNORE_INDEX, and cross-entropy already
skips them.

FROZEN AT CONSTRUCTION, on purpose. A val set that is re-sampled every time it is
used is a ruler that moves, and two runs measured with it are not comparable. The
cost is holding a few hundred rows of integers; the alternative has silently
invalidated comparisons in this repo before.

THE AGGREGATION IS TOKEN-WEIGHTED, not mean-of-batch-means: total nats over all
supervised positions, divided by the count of them. Two functionals with the same
name are not the same metric — a batch-mean and a pooled mean differ whenever rows
carry different numbers of supervised tokens, which is exactly the t2m case
(captions vary in length). Written down because the difference is invisible in the
number and only shows up when someone compares against a run that aggregated the
other way.

Scheduling is core's Evaluator contract: the Trainer asks should_eval(step) every
step. The default is periodic; scaling.py passes an explicit log-spaced eval_at.
"""

import torch
import torch.nn.functional as F

from core.evaluation.evaluator import Evaluator
from core.tokenization.vocab_layout import VocabLayout


class SupervisedCE(Evaluator):
    """Args:
        loader:     a val-split MixedDataLoader (assembly.build_loader(..., "val"))
        n_batches:  how many batches to freeze
        metric:     metric name, e.g. "val/video_ce"
    """

    def __init__(self, loader, n_batches=20, interval_steps=500, eval_at=None,
                 metric="val/ce", label=""):
        self.interval_steps = interval_steps
        self.eval_at = {int(s) for s in eval_at} if eval_at else None
        self.metric = metric
        self.label = label
        self.best = float("inf")

        it = iter(loader)
        self._batches = []
        n_sup = 0
        for _ in range(n_batches):
            b = next(it)
            # Keep only what the forward needs. `state_dict` holds live source
            # objects; carrying it would pin the whole loader alive for the run.
            kept = {k: b[k] for k in ("idx", "token_types", "targets") if k in b}
            self._batches.append(kept)
            n_sup += int((kept["targets"] != VocabLayout.IGNORE_INDEX).sum())
        self.n_supervised = n_sup

    def describe(self) -> str:
        """One line for the startup log: WHICH rows, HOW many, HOW often. Both
        rulers must speak in the log — the train/val drift bug survived for months
        because only the training side did."""
        when = (f"at steps {sorted(self.eval_at)[:4]}..." if self.eval_at
                else f"every {self.interval_steps} steps")
        return (f"{self.metric}: {len(self._batches)} frozen val batches "
                f"({self.n_supervised:,} supervised tokens){self.label}, {when}")

    @torch.no_grad()
    def evaluate(self, system, autocast_ctx) -> dict:
        total_nats, total_tokens = 0.0, 0
        with autocast_ctx:
            for b in self._batches:
                hidden = system.trunk(b["idx"], token_types=b.get("token_types"))
                logits = system.head(hidden)
                tgt = b["targets"]
                nats = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                    ignore_index=VocabLayout.IGNORE_INDEX, reduction="sum")
                total_nats += float(nats)
                total_tokens += int((tgt != VocabLayout.IGNORE_INDEX).sum())
        mean = total_nats / max(total_tokens, 1)
        self.best = min(self.best, mean)
        return {self.metric: mean, f"{self.metric}_best": self.best}
