"""
serve/browse.py — the data browser's backend. PLAN.md "数据浏览器" is the screen
design; this file is its contract.

THE ONE RULE. Every batch this panel shows comes from assembly.build_loader(...) —
the SAME call train.py makes. It never opens a cache file itself. That is not
tidiness: training and evaluation in this repo once assembled sequences by two
different paths and disagreed for months with nothing in the log to show it. "What
you are looking at is what the model is fed" has to be structural, or it is a
promise that quietly stops being true.

Endpoints, one per zoom level:

    vocab(line)          the band table: the canonical five-band layout and THIS
                         line's compact one, side by side
    batch(line, split)   one real batch as per-row band spans + supervised runs
    row(line, i)         one row, fully dissected (spans + units + media)
    token(line, i, pos)  one token: id, band, local id, type, what it means
    stats(line, split)   frequency by band, supervised fraction, row lengths

Everything except row() is cheap and needs no GPU: the band map and the supervision
map come from token_types and targets, which the loader already produced. Only
actually LOOKING at a row costs a decode.

State: one assembled (vocab, loader, batch) per (line, split), built on first use
and kept. Assembling the t2m source tokenizes 30k captions, so rebuilding it per
request would make the panel unusable — and a batch that changed under the user
between the overview and the row view would be worse than slow.
"""

import numpy as np

from projects.nano_multimodal import assembly, decode, spec
from projects.nano_multimodal.assembly import TYPE_IDS
from projects.nano_multimodal.decode import band_of

_STATE = {}


def state(line, split="val"):
    """The assembled (vocab, batch) for a line, built once."""
    key = (line, split)
    if key not in _STATE:
        config = assembly.load_config(line)
        vocab = assembly.assemble_vocab(line, config)
        # The browser shows ONE batch. Reading the whole corpus to get it costs 37 s
        # on the motion line (every caption BPE-encoded up front), paid on the first
        # click and again on every restart. A cap keeps the rows real and the panel
        # responsive; training is untouched, which is the only place the full corpus
        # matters.
        for sc in config["data"].get("sources", []) + config["data"].get("val_sources", []):
            sc.setdefault("limit", BROWSE_CLIPS[0])
        loader = assembly.build_loader(line, config, vocab, split, DEVICE[0],
                                       batch_size=BATCH[0])
        batch = next(iter(loader))
        _STATE[key] = {"vocab": vocab, "batch": batch, "config": config}
    return _STATE[key]


DEVICE = ["cuda"]      # set by app.py --device
BATCH = [8]
BROWSE_CLIPS = [2000]  # clips read per line for the browser (see state())


def lines():
    """Which lines this server can show, and why not, when it cannot."""
    out = []
    for line in ("text", "motion", "video"):
        try:
            assembly.assemble_vocab(line, assembly.load_config(line))
            out.append({"line": line, "ok": True})
        except Exception as e:                              # noqa: BLE001
            out.append({"line": line, "ok": False, "why": f"{type(e).__name__}: {e}"})
    return {"lines": out}


def vocab(line):
    """THIS line's vocabulary: which bands it has and where each one starts.

    One table, not two. The five-band canonical layout used to be drawn above it so a
    student could see that offsets are assembled rather than assigned — and it cost
    more confusion than it bought, because the two bars invited the question "which
    one is real?" before the first one had been understood. The point survives as one
    sentence in the caption; the picture is of the vocabulary this model actually has.

    Drawn as a SCHEMATIC, not to scale. Proportionally, a text band of 32750 and a
    control band of 18 differ by three orders of magnitude, so the honest picture is
    one that shows nothing at all about the small band. The numbers are written on the
    segments instead, which is where a reader looks for them anyway.
    """
    v = state(line)["vocab"]
    bands = []
    for n in spec.BAND_ORDER:
        t = TYPE_IDS[n]
        if t in v.layout.ranges:
            lo, hi = v.layout.ranges[t]
            bands.append({"band": n, "type_id": t, "start": lo, "end": hi,
                          "size": hi - lo})
    return {"line": line, "bands": bands, "vocab_size": v.layout.vocab_size,
            "n_token_types": v.layout.n_token_types, "sequence_len": v.sequence_len}


def _runs(mask):
    """[bool] -> [[start, end), ...]. The supervision map is nearly all runs, so
    sending runs instead of 1300 booleans per row keeps the batch view small."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append([start, i]); start = None
    if start is not None:
        out.append([start, len(mask)])
    return out


def rows(line, split="val"):
    """A one-line summary per row, for the picker. The batch heat-map this replaces
    tried to show sixteen rows at once and ended up showing none of them clearly;
    choosing a row and reading it properly is the thing worth doing."""
    st = state(line, split)
    v, b = st["vocab"], st["batch"]
    idx = b["idx"].cpu().numpy()
    tgt = b["targets"].cpu().numpy()
    out = []
    for r in range(idx.shape[0]):
        label = f"行 {r}"
        if "text" in v.active:                      # a caption or the opening words
            tok = v.tokenizers.get("text")
            t_lo, t_hi = v.layout.ranges[TYPE_IDS["text"]]
            ids = [int(x) for x in idx[r] if t_lo <= int(x) < t_hi][:14]
            if tok and ids:
                label += " · " + tok.decode(ids).strip().replace("\n", " ")[:60]
        out.append({"row": r, "label": label,
                    "n_supervised": int((tgt[r] != -1).sum())})
    return {"line": line, "split": split, "seq_len": int(idx.shape[1]), "rows": out}


def row(line, i, split="val", **kw):
    """One row, dissected into what the browser draws: a flow of CELLS.

    Each cell is one token and carries two things — the integer the model sees, and
    the human-readable piece it stands for. That pairing is the whole screen: a
    student reads a sentence, or a run of frames, and directly under each piece is the
    integer that produced it.

    A cell also says whether its integer is one the loss is computed on. The panel
    colours those differently, which replaces a separate "target" track that said the
    same thing in a vocabulary nobody outside this repo shares.

    Cells whose readable piece is not per-token — 256 video codes are one frame, 28
    motion codes are one clip — are grouped: the group carries a NAME, the name is
    shown once over the run, and the media it names is played underneath. So the row
    reads `[motion 1]` above and its integers below, and the animation for `[motion 1]`
    sits below the row.

    The media comes from decode.render — the SAME call the inference panel makes on a
    row the model just wrote. Any difference a student sees between the two panels is
    therefore the model, never the renderer.
    """
    st = state(line, split)
    v, b = st["vocab"], st["batch"]
    idx = b["idx"].cpu().numpy()
    tgt = b["targets"].cpu().numpy()
    i = max(0, min(int(i), idx.shape[0] - 1))
    ids = idx[i].tolist()
    r = decode.render(line, ids, v, device=DEVICE[0], **kw)
    sup = (tgt[i] != -1).tolist()

    # A padding tail is one fact, not two hundred. Runs of the SAME control token
    # collapse to a single cell carrying its length: without this, 199 cells of `eos`
    # push the sentence and its motion off the top of the screen, and the screen is
    # about the sentence.
    RUN = 6
    collapse = {}
    j = 0
    while j < len(ids):
        k = j
        while k + 1 < len(ids) and ids[k + 1] == ids[j] and r.units[j].get("band") == "control":
            k += 1
        if r.units[j].get("band") == "control" and k - j + 1 >= RUN:
            collapse[j] = k - j + 1
            for x in range(j + 1, k + 1):
                collapse[x] = 0                       # 0 = swallowed by the run head
        j = k + 1

    cells, groups = [], []
    for pos, (tid, u) in enumerate(zip(ids, r.units)):
        rep = collapse.get(pos)
        if rep == 0:
            continue
        band = u.get("band")
        text = u.get("preview") or ""
        group = None
        if band in ("video", "motion"):
            # One media group per latent frame (video) or per clip (motion).
            key = u.get("frame", 0) if band == "video" else 0
            group = f"{band}:{key}"
            if not groups or groups[-1]["key"] != group:
                groups.append({"key": group, "band": band, "index": len(groups),
                               "name": (f"[frame {key}]" if band == "video"
                                        else "[motion 1]"),
                               "start": pos, "end": pos + 1})
            else:
                groups[-1]["end"] = pos + 1
            text = ""
        cell = {"pos": pos, "id": int(tid), "band": band, "text": text,
                "sup": bool(sup[pos]), "group": group}
        if rep:
            cell["repeat"] = rep
        cells.append(cell)
    return {"line": line, "row": i, "kind": r.kind, "cells": cells, "groups": groups,
            "units": r.units, "notes": r.notes, "n_supervised": int(sum(sup)),
            "n_tokens": len(ids), "media": r.media}


def token(line, pos, i=0, split="val"):
    """One token: what it is, and what it means. The unit is what carries the
    modality-specific answer — a word-piece, four frames of movement, or one 8x8
    pixel patch."""
    st = state(line, split)
    v, b = st["vocab"], st["batch"]
    idx = b["idx"].cpu().numpy()
    tgt = b["targets"].cpu().numpy()
    i, pos = int(i), int(pos)
    tid = int(idx[i][pos])
    band, type_id, local = band_of(v, tid)
    return {"position": pos, "id": tid, "band": band, "type_id": type_id,
            "local_id": local,
            "control_name": v.resolver.name_of(tid),
            "supervised": bool(tgt[i][pos] != -1),
            "predicts": int(tgt[i][pos]) if tgt[i][pos] != -1 else None}


def stats(line, split="val"):
    """Cheap answers to 'is there enough data, and is it balanced': token counts by
    band, the supervised fraction, and — on the video line — the action histogram.

    On the motion line this is where the overfitting lesson gets its premise: one
    epoch of Bones-SEED is 450,594 rows = 30.4M supervised motion tokens, against a
    Chinchilla appetite of ~800M for the 40M-parameter model that reads them."""
    st = state(line, split)
    v, b = st["vocab"], st["batch"]
    idx = b["idx"].cpu().numpy().ravel()
    tgt = b["targets"].cpu().numpy()
    types = v.layout.classify_token_types(b["idx"].cpu()).numpy().ravel()
    by_type = {t: n for n, t in TYPE_IDS.items()}
    counts = {}
    for t, c in zip(*np.unique(types, return_counts=True)):
        counts[by_type.get(int(t), f"type{int(t)}")] = int(c)

    out = {"line": line, "by_band": counts,
           "supervised": int((tgt != -1).sum()), "total": int(tgt.size)}
    if "action" in v.active:
        a_lo, a_hi = v.layout.ranges[TYPE_IDS["action"]]
        acts = idx[(idx >= a_lo) & (idx < a_hi)] - a_lo
        hist = np.bincount(acts, minlength=spec.N_ACTIONS).tolist()
        out["actions"] = [{"id": i, "name": spec.ACTION_NAMES[i], "count": int(c)}
                          for i, c in enumerate(hist)]
    if "video" in v.active:
        v_lo, v_hi = v.layout.ranges[TYPE_IDS["video"]]
        codes = idx[(idx >= v_lo) & (idx < v_hi)] - v_lo
        out["code_vocab"] = {"used": int(len(np.unique(codes))),
                             "total": int(v_hi - v_lo), "drawn": int(len(codes))}
    return out
