"""
sources.py — 世界 -> token. The encode side of the boundary.

Its mirror image is decode/ (token -> 世界). Between them sits the model, which
never sees anything but integers. That symmetry is the project, and it is meant to
be visible in the directory listing.

TWO of the three sources already exist and are used unchanged:

    text    modalities.text.TextDataSource      FineWeb parquet, BPE on the fly
    motion  modalities.motion...T2MDataSource   pre-encoded (caption, codes) pairs

ONE is new, and it is the only data code this project writes:

    video   VideoRowSource                      pre-encoded fixed-stride memmap

All three satisfy core's DataSource contract — an infinite iterator of
{tokens, token_types, attention_mask, loss_weights}, all [S] and padded — so
MixedDataLoader mixes, stacks, shifts and checkpoints all three identically. The
orchestrator passes SOURCE_TYPES as a LOCAL mapping, never a global registry
(vision.md, "Config-driven switching is an orchestrator concern").

A consequence worth knowing: because all three are ordinary DataSources, listing
more than one under `data.sources` in a config trains ONE model on several
modalities at once. That is a later lesson, not this one, but it costs no code.
"""

from typing import Any, Dict, Iterator, Optional

import torch

from core.data.data_source import DataSource


class VideoRowSource(DataSource):
    """Pre-encoded VizDoom clips -> one flat token row each.

    The cache is two flat binaries plus a meta.json (built offline; students are
    given it): `{split}_codes.u16` of code_len ids per row and `{split}_actions.u8`
    of one action id per game frame. Every row of a given clip length has the same
    width, so `row i` is a slice and random access costs a page fault — which is
    what lets core's ResumableDistributedSampler drive it, and what makes a run
    checkpointed on 2 GPUs resume on 1 (its state is {seed, epoch, index}: three
    integers with no rank in them).

    THE ROW. Codes and actions are interleaved into one flat sequence, then wrapped
    by the recipe exactly the way text and motion are wrapped:

        [bos, video_start, L0, a a a a, L1, a a a a, ..., L4, video_end, eos]
               ^ 256 codes per latent frame        ^ td=4 action ids per frame

    There is no RowLayout, no block mask and no 3D rope: this is a plain causal
    sequence, shifted by one, identical in every respect to the text line. The
    interleaving is a property of the DATA, not of the model.

    SUPERVISION. loss_weights is 0 on:
      * the action tokens — they are the control INPUT; learning to predict which
        button a player presses is a different problem, and supervising it spends
        capacity on it;
      * the codes of L0 — frame 0 is the GIVEN observation, not a prediction.
    and 1 on the remaining code positions plus the closing tag. That is the whole
    mechanism: MixedDataLoader turns weight-0 positions into IGNORE_INDEX targets,
    and the fused cross-entropy already honours ignore_index. The motion line
    expresses "supervise the motion half only" through the identical field.
    """

    def __init__(self, config: Dict[str, Any], tokenizers: Dict):
        import json
        from pathlib import Path

        import numpy as np

        from core.data.dist_sampler import ResumableDistributedSampler
        from core.data.sequence_recipe import SequenceRecipe

        from projects.nano_multimodal import spec

        self._layout = tokenizers["layout"]
        self._resolver = tokenizers["control_resolver"]
        self.split = config.get("split", "train")
        self.sequence_len = int(config["sequence_len"])

        cache = Path(config["cache"])
        meta = json.loads((cache / "meta.json").read_text())
        contract = meta.get("shape_contract", meta.get("geometry"))
        n_rows = meta["rows"][self.split]
        self.contract = contract
        self.codes = np.memmap(cache / f"{self.split}_codes.u16",
                               dtype=meta["code_dtype"], mode="r",
                               shape=(n_rows, contract["code_len"]))
        self.actions = np.memmap(cache / f"{self.split}_actions.u8",
                                 dtype=meta["action_dtype"], mode="r",
                                 shape=(n_rows, contract["n_action_tokens"]))

        cpf, n_lat, td = contract["codes_per_frame"], contract["n_latent"], contract["td"]
        n_blocks = n_lat - 1
        self.cpf, self.n_lat, self.td, self.n_blocks = cpf, n_lat, td, n_blocks

        # --- the interleave, laid out ONCE ---------------------------------
        # [L0, a*td, L1, a*td, ..., L4] as one flat field. Ported slot-for-slot
        # from exemplars/nano_world_model/row_layout.py: which action ids sit
        # between which frames is a fact of how the CACHE was built, and getting
        # it wrong misaligns the conditioning without misaligning anything a test
        # would notice.
        field_len = n_lat * cpf + n_blocks * td
        code_slots, action_slots, predicted = [], [], []
        p = 0
        for k in range(n_lat):
            if k > 0:                       # frame 0 is the given observation
                action_slots.extend(range(p, p + td))
                p += td
                predicted.extend(range(p, p + cpf))
            code_slots.extend(range(p, p + cpf))
            p += cpf
        assert p == field_len
        self._code_slots = np.asarray(code_slots)      # cache column order
        self._action_slots = np.asarray(action_slots)
        # The last cache action drives a frame past the window; only n_blocks*td used.
        self._action_cols = np.arange(n_blocks * td)

        recipe = SequenceRecipe(
            template=config["recipe"]["template"],
            supervise=config["recipe"].get("supervise", "all"),
            supervise_tags=config["recipe"].get("supervise_tags"),
            constants=config["recipe"].get("constants"),
        )
        self.recipe_name = config.get("recipe_name", "<inline>")
        self.v_off = self._layout.offset(spec.VIDEO_TYPE_ID)
        self.a_off = self._layout.offset(spec.ACTION_TYPE_ID)
        self.n_actions = spec.N_ACTIONS

        fixed = recipe.build_fixed_layout(
            {"video_tokens": field_len}, self._layout, self._resolver,
            field_dummy_ids={"video_tokens": self.v_off})
        row_len = int(fixed["token_template"].numel())
        if row_len != self.sequence_len:
            raise ValueError(
                f"video row is {row_len} tokens but sequence_len={self.sequence_len}. "
                f"It is DERIVED from the clip (spec.video_shape), not configured — "
                f"leave sequence_len: null in the yaml.")
        f0, f1 = fixed["field_slices"]["video_tokens"]
        self._field = (f0, f1)

        # --- supervision, written once -------------------------------------
        # 1 on the code positions of the PREDICTED latent frames, plus the closing
        # video_end tag; 0 on the given frame, on every action token, and on the
        # delimiters. Ported from exemplars/nano_world_model/autoregressive.py,
        # which states the reasons: the given frame is the observation, and the
        # actions are the control INPUT — learning which button a player presses is
        # a different problem that would spend capacity on it.
        lw = np.zeros(row_len, dtype=np.float32)
        lw[f0 + np.asarray(predicted)] = 1.0
        lw[f1] = 1.0                                   # video_end
        self.n_supervised = int(lw.sum())
        assert self.n_supervised == n_blocks * cpf + 1

        dev = config.get("device", "cuda")
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        self.device = torch.device(dev)
        self._template = fixed["token_template"].numpy().copy()
        self._types = fixed["token_types"].to(self.device)
        self._attn = torch.ones(row_len, dtype=torch.long, device=self.device)
        self._lw = torch.from_numpy(lw).to(self.device)

        self._sampler = ResumableDistributedSampler(self.codes,
                                                    seed=int(config.get("seed", 0)))
        rs = config.get("resume_state")
        if rs:
            self._sampler.load_state_dict(rs)

        print(f"VideoRowSource[{self.split}]: {n_rows} rows x {contract['code_len']} "
              f"codes, {self.n_supervised} supervised/row, recipe[{self.recipe_name}], "
              f"cache={cache.name}")

    def _row(self, i):
        """Cache row -> one flat token row of GLOBAL ids."""
        import numpy as np
        row = self._template.copy()
        f0 = self._field[0]
        row[f0 + self._code_slots] = self.v_off + np.asarray(self.codes[i], dtype=np.int64)
        a = np.clip(np.asarray(self.actions[i], dtype=np.int64)[self._action_cols],
                    0, self.n_actions - 1)
        row[f0 + self._action_slots] = self.a_off + a
        return row

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        import numpy as np
        while True:
            for i in self._sampler:
                yield {
                    "tokens": torch.from_numpy(self._row(i)).to(self.device),
                    "token_types": self._types,        # shared (read-only)
                    "attention_mask": self._attn,      # shared (read-only)
                    "loss_weights": self._lw,          # shared (read-only)
                }

    def get_state(self) -> Optional[Dict[str, Any]]:
        return self._sampler.state_dict()

    def set_state(self, state: Dict[str, Any]) -> None:
        self._sampler.load_state_dict(state)

    def __repr__(self) -> str:
        s = self._sampler.state_dict()
        return f"video:(ep={s['epoch']}, i={s['index']})"


def source_types() -> Dict[str, type]:
    """The local string->class mapping MixedDataLoader resolves `type:` against.

    A LOCAL parameter, not a global registry — vision.md's "config-driven switching
    is an orchestrator concern", and its cautionary example is a project that
    mutated a framework-level global to get exactly this.

    Imports are lazy and failures are tolerated per entry, so a text-only run never
    needs the motion package (numpy converters, and the SMPL body model on the
    decode side) and a motion run never needs the video cache reader.
    """
    types: Dict[str, type] = {}
    try:
        from modalities.text import TextDataSource
        types["text"] = TextDataSource
    except Exception:                                    # noqa: BLE001 - optional line
        pass
    try:
        from modalities.motion.data.sources import MotionDataSource, T2MDataSource
        types["t2m"] = T2MDataSource
        types["motion"] = MotionDataSource
    except Exception:                                    # noqa: BLE001 - optional line
        pass
    types["video"] = VideoRowSource
    return types
