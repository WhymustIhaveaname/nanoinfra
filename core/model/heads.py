"""
Pluggable output heads.

A head owns the un-embedding parameters and the loss/logits computation, as a
SEPARATE nn.Module from the trunk (the GPT body). This buys three things:
  - the orchestrator assembles the head (it is not baked into the model);
  - the head gets its own FSDP shard group — params are lazily all-gathered and
    every head call is a legitimate FSDP forward window (no more window-outside
    calls, which the old GPT.head_* methods relied on FSDP-root-no-reshard to
    survive);
  - torch.compile treats trunk and head as independent frames.

Behavior families (naive CE / Liger fused CE / future band-factorized) are chosen by
`__class__` INJECTION at assembly time (`XXXHead.setup(head)`), BEFORE `fully_shard`,
once. Runtime never changes `__class__`.

One family is NOT a class: head_ce="compiled" keeps the naive class and binds a
compiled `loss` as an INSTANCE attribute instead — `torch.compile` produces a
callable, not a type, so there is nothing to inject. That asymmetry is why
`type_losses` below reaches for `type(self).loss` rather than `self.loss`.

Entry points that touch head params outside the trunk forward
(loss / logits / type_losses) are registered as FSDP forward methods AFTER shard.

FSDP2 timing rule (why the order matters): the `__class__` injection MUST happen
BEFORE `fully_shard` — a post-shard `__class__` swap silently drops FSDP's dynamically
mixed-in class — and `register_fsdp_forward_method` MUST happen AFTER, since it no-ops
on a non-FSDPModule.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.tokenization.vocab_layout import VocabLayout

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except Exception:                       # noqa: BLE001 — deliberately broad, see below
    # NOT just ImportError. `liger_kernel.transformers` runs @triton.autotune at
    # import time, which wants a live Triton driver; on a machine with none — a
    # cluster login node, say — it raises `RuntimeError: 0 active drivers`, which
    # sails straight through an ImportError guard and takes `import core.model.heads`
    # down with it. Reported from a CentOS 7 / A100 deployment, 2026-08-26.
    #
    # A broad except at an optional-import site is only safe because of what the
    # flag now means downstream: head_ce="liger" RAISES when this is False, it does
    # not quietly become another arm. Before that was true, swallowing everything
    # here would have meant a login-node quirk silently changing which kernel a
    # training run used.
    LigerFusedLinearCrossEntropyLoss = None

LIGER_AVAILABLE = LigerFusedLinearCrossEntropyLoss is not None


class LMHead(nn.Module):
    """Linear un-embedding + softcap; naive (F.cross_entropy) loss path.

    Tensor-level building block: methods take (hidden, targets), NOT a batch dict.
    Batch unpacking lives one level up in LMSystem. All three entry points below are
    called OUTSIDE the trunk forward, so under FSDP each must be registered via
    register_fsdp_forward_method (done in model_setup after shard).
    """

    def __init__(self, n_embd, vocab_size, softcap=15.0):
        super().__init__()
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.softcap = softcap

    def init_weights(self):
        # matches the old GPT.init_weights: the classifier starts at zero.
        torch.nn.init.zeros_(self.lm_head.weight)

    def forward(self, hidden):
        """hidden [B,T,H] -> softcapped logits [B,T,V] (fp32). Eval / inference entry."""
        logits = self.lm_head(hidden).float()
        return self.softcap * torch.tanh(logits / self.softcap)

    def loss(self, hidden, targets):
        """CE loss (scalar). Training entry. Naive path materializes logits."""
        logits = self.forward(hidden)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=VocabLayout.IGNORE_INDEX,
            reduction="mean",
        )

    def type_losses(self, hidden, targets, target_types, type_ids):
        """Per-type CE losses {type_id: scalar} via masked targets (eval).

        Calls `type(self).loss(self, ...)`, not `self.loss(...)` and not
        `LMHead.loss(self, ...)`. Both details are load-bearing:

          - `self.loss` would pick up whatever is bound as an INSTANCE attribute.
            Two things bind one: head_ce="compiled" (a dynamo wrapper) and, under
            FSDP, register_fsdp_forward_method. Reaching either from in here nests a
            frame or a forward window inside the one type_losses is already running
            in — untested, and this is an eval path not worth the risk.
          - `LMHead.loss` would hard-bind the naive implementation, silently
            downgrading a LigerLMHead's per-type eval to the unfused path (and
            materializing [B,T,V] logits once per type id, which is the exact cost
            liger exists to remove).

        `type(self)` threads that needle: it skips the instance dict but still
        resolves through the MRO, so a liger head gets liger — including after
        fully_shard rewrites __class__ to an FSDP-mixed subclass.
        """
        result = {}
        for tid in type_ids:
            masked = torch.where(
                target_types == tid, targets,
                torch.full_like(targets, VocabLayout.IGNORE_INDEX),
            )
            result[tid] = type(self).loss(self, hidden, masked)
        return result


class LigerLMHead(LMHead):
    """Liger fused-CE loss path (memory-safe: never materializes [B,T,V] logits).

    Selected by injection: `LigerLMHead.setup(head)` prepares `fused_ce` and swaps
    `__class__`. MUST run BEFORE `fully_shard(head)`. Only the training `loss()` is
    replaced; `forward()`/`type_losses()` inherit LMHead's logits path.
    """

    @classmethod
    def setup(cls, head):
        assert LigerFusedLinearCrossEntropyLoss is not None, "liger_kernel not installed"
        head.fused_ce = LigerFusedLinearCrossEntropyLoss(
            ignore_index=VocabLayout.IGNORE_INDEX, reduction="mean", softcap=head.softcap,
        )
        head.__class__ = cls  # inject: swap the method table; __dict__ (params, fused_ce) preserved

    def loss(self, hidden, targets):
        if hidden.device.type == "cuda":
            return self.fused_ce(
                self.lm_head.weight,
                hidden.reshape(-1, hidden.size(-1)),
                targets.reshape(-1),
            )
        return LMHead.loss(self, hidden, targets)  # cpu fallback
