"""
build_video_cache.py — a CLASSROOM-SIZED slice of the main-line video corpus.

    python -m projects.nano_multimodal.data.build_video_cache
    python -m projects.nano_multimodal.data.build_video_cache --rows 288000

The exemplar ships a 4,733-row smoke corpus: enough to prove the wiring, not enough
to train a model worth looking at. The main line (datasets/pipe4) holds 8.1M frames
= 880k rows at this contract, which is more than an hour of GPU. This cuts the
middle: a subset sized so that ONE HOUR of training is ONE PASS over it.

WHY ONE PASS. A student's first world model should be short of DATA, not short of
epochs. Repeating a small corpus produces a model that looks better than it is and
teaches the wrong lesson about why it is bad.

    288,000 rows x 1300 tokens / (32 rows per step) = 9,000 steps
    9,000 steps x 0.40 s/step (measured, RTX 5090, dbs 16, compiled) = 1.0 hour

WHY STRIDED, NOT THE FIRST N SHARDS. The corpus is recorded in LAYERS (coverage,
events, forks, long runs) and shards are written in recording order, so a prefix is a
biased sample of one part of the recipe. Taking every k-th shard spans the whole
corpus for the same cost. It is also deterministic, so two students who build the
subset get the same one — which is what makes their numbers comparable.

Writes the format VideoRowSource reads: one flat binary per field plus a meta.json,
identical to what exemplars/nano_world_model/build_cache.py produces. The per-shard
assertions are ported from there rather than re-derived: a codec swap that widened
the codebook past 65535 would otherwise wrap silently and poison the cache.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from projects.nano_multimodal import spec

CODE_DTYPE = np.uint16          # code ids
ACTION_DTYPE = np.uint8         # action ids

SOURCE = spec.REPO / "datasets" / "pipe4"
DEFAULT_TRAIN_ROWS = 288_000    # ~1 hour at 32 rows/step; see the module docstring
DEFAULT_VAL_ROWS = 8_000


def shards(frames, split):
    """Encoded shards of this clip length, sorted. THE ORDER IS THE CACHE'S ROW ORDER
    and must not vary between machines."""
    suffix = f"_{frames}f" + ("_val" if split == "val" else "")
    out = sorted((SOURCE / "codes").glob(f"*{suffix}.npz"))
    # `*_17f.npz` also matches `*_17f_val.npz`, so the train list must exclude them.
    if split == "train":
        out = [p for p in out if not p.name.endswith("_val.npz")]
    return out


def pick(paths, budget, rows_of):
    """Stride across the corpus until the row budget is met. Returns (paths, rows)."""
    if not paths:
        return [], 0
    total = sum(rows_of(p) for p in paths)
    if total <= budget:
        return paths, total
    step = max(1, round(total / budget))
    chosen, got = [], 0
    for p in paths[::step]:
        if got >= budget:
            break
        chosen.append(p); got += rows_of(p)
    i = 0
    while got < budget and i < len(paths):        # top up if the stride overshot
        if paths[i] not in chosen:
            chosen.append(paths[i]); got += rows_of(paths[i])
        i += 1
    chosen.sort()
    return chosen, got


def stream(paths, out_dir, name, contract):
    """Stream shards into <name>_codes.u16 / <name>_actions.u8."""
    cp, ap = out_dir / f"{name}_codes.u16", out_dir / f"{name}_actions.u8"
    prov, total, t0 = [], 0, time.time()
    with open(cp, "wb") as fc, open(ap, "wb") as fa:
        for i, p in enumerate(paths):
            d = np.load(p, allow_pickle=True)
            codes, acts = d["codes"], d["actions"]
            assert int(d["frames"]) == contract["frames"], f"{p.name}: wrong clip length"
            assert int(d["res"]) == contract["res"], f"{p.name}: wrong resolution"
            assert str(d["tokenizer"]) == spec.CODEC_NAME, \
                f"{p.name}: made by {d['tokenizer']}, spec says {spec.CODEC_NAME}"
            assert codes.shape[1] == contract["code_len"], \
                f"{p.name}: code_len {codes.shape[1]} != contract {contract['code_len']}"
            assert acts.shape[1] == contract["n_action_tokens"], \
                f"{p.name}: {acts.shape[1]} action tokens != {contract['n_action_tokens']}"
            assert len(codes) == len(acts), f"{p.name}: codes/actions row mismatch"
            assert codes.min() >= 0 and codes.max() < np.iinfo(CODE_DTYPE).max, \
                f"{p.name}: code id outside uint16 (did the codebook grow?)"
            assert acts.min() >= 0 and acts.max() < spec.N_ACTIONS, \
                f"{p.name}: action id outside 0..{spec.N_ACTIONS - 1}"
            atv = int(d["action_table_version"]) if "action_table_version" in d.files else 1
            assert atv <= spec.ACTION_TABLE_VERSION, \
                f"{p.name}: action table v{atv} is newer than spec v{spec.ACTION_TABLE_VERSION}"
            fc.write(np.ascontiguousarray(codes, dtype=CODE_DTYPE).tobytes())
            fa.write(np.ascontiguousarray(acts, dtype=ACTION_DTYPE).tobytes())
            prov.append({"shard": p.stem, "rows": int(len(codes))})
            total += len(codes)
            if i % 20 == 0 or i == len(paths) - 1:
                print(f"  [{name}] {i+1}/{len(paths)} shards, {total:,} rows, "
                      f"{time.time()-t0:.0f}s", flush=True)
    return total, prov


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--rows", type=int, default=DEFAULT_TRAIN_ROWS)
    ap.add_argument("--val-rows", type=int, default=DEFAULT_VAL_ROWS)
    ap.add_argument("--frames", type=int, default=spec.FRAMES)
    ap.add_argument("--res", type=int, default=spec.RES)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    contract = spec.video_shape(args.frames, args.res)
    contract["n_action_tokens"] = args.frames          # the cache stores one per game frame
    out = Path(args.out) if args.out else spec.video_cache_dir(args.frames, args.res)
    out.mkdir(parents=True, exist_ok=True)

    if not (SOURCE / "codes").is_dir():
        raise SystemExit(f"no {SOURCE/'codes'} — this cuts a slice of the main-line "
                         f"corpus, which has to be on disk already")

    # Row counts come from the MANIFEST, not from opening each npz: reading the shape
    # decompresses the whole array, and there are hundreds of shards. The manifest is
    # what encode.py recorded about them, which is the right source anyway.
    counts = {}
    with open(SOURCE / "manifest.jsonl") as f:
        for ln in f:
            e = json.loads(ln)
            for split_key, n in (e.get("rows") or {}).items():
                counts[f"{e['shard']}_{split_key.replace('_train','').replace('_val','')}"
                       + ("_val" if split_key.endswith("_val") else "")] = int(n)

    def rows_of(path):
        n = counts.get(path.stem)
        if n is None:                      # a shard the manifest does not describe
            n = int(np.load(path, allow_pickle=True)["codes"].shape[0])
            counts[path.stem] = n
        return n
    print(f"source {SOURCE}, contract {contract['frames']}f/{contract['res']}px, "
          f"row = {contract['row_len']} tokens")
    meta = {"shape_contract": contract, "codec": spec.CODEC_NAME,
            "code_dtype": "uint16", "action_dtype": "uint8",
            "source": {"root": str(SOURCE)}, "rows": {}}
    for split, budget in (("train", args.rows), ("val", args.val_rows)):
        allp = shards(args.frames, split)
        print(f"{split}: {len(allp)} shards available")
        chosen, want = pick(allp, budget, rows_of)
        print(f"  taking {len(chosen)} of them (stride), ~{want:,} rows for a "
              f"{budget:,}-row budget")
        n, prov = stream(chosen, out, split, contract)
        meta["rows"][split] = n
        meta["source"][split] = prov
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    mb = sum(f.stat().st_size for f in out.glob("*.u*")) / 2**20
    print(f"\nwrote {out}  —  train {meta['rows']['train']:,} / "
          f"val {meta['rows']['val']:,} rows, {mb:.0f} MiB")
    steps = meta["rows"]["train"] // 32
    print(f"  at 32 rows/step that is {steps:,} steps = one pass "
          f"(~{steps*0.40/3600:.1f} h on an RTX 5090 at the measured 0.40 s/step)")


if __name__ == "__main__":
    main()
