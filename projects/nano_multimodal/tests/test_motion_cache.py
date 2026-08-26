"""
test_motion_cache.py — does this motion cache actually pair with this codec?

The tag cannot answer that. Every tagged cache in outputs/motion_caches names the
same codec, and only some of them decode to a human. Codes from two different
preparations are the same integers on disk and look identical, which is the failure
REGISTRY.md warns about — and a warning is all a string can give you.

So this asks the poses. Decode real rows from the cache the config names and check
the body STANDS UP: mean per-frame extent along the up axis, over the joints. A
human is ~1.5-1.9 m tall; a mispaired cache decodes to something under ~1.2 m,
because the poses are subtly scrambled rather than obviously broken.

Measured on this repo, all through spec.MOTION_CODEC:

    bones_seed ground truth, encode->decode round trip   1.47 -> 1.47 m
    t2m_bones_{train,val}.npz                            1.48 m   pairs
    bones_seed_{train,val}_codes_k512_canon.npz          1.48 m   pairs
    t2m_humanml3d_train.npz                              0.82 m   DOES NOT
    t2m_humanml3d_val.npz                                1.01 m   DOES NOT
    lafan1_{train,val}_codes_k512.npz                    1.03 / 1.10 m   DOES NOT
    t2m_{train,val}.npz (UNTAGGED)                       1.41 / 1.21 m   borderline

The last row is the reason this check is a floor and not a verdict: those two caches
sit just above the threshold on eight clips and just below it on six. They are also
the untagged ones, so they are refused earlier, by the tag rule in
assembly.motion_cache_facts — which is the belt to this check's braces. A number
near 1.2 means "look at it", not "ship it".

The codec is not the problem. Something about how the other caches were prepared
before encoding differs — note that the bones code files are named `..._canon.npz`
and the others are not — and no sidecar field records it.

    python -m projects.nano_multimodal.tests.test_motion_cache
    python -m projects.nano_multimodal.tests.test_motion_cache --survey   # all caches
"""

import argparse
import glob
import os

import numpy as np

from projects.nano_multimodal import assembly, spec

UPRIGHT_M = 1.20        # below this, the body is not standing; see the table above
SMOOTH = 1.50           # jitter above this and the motion is unwatchable, however
                        # upright the poses are. Raw Bones-SEED sits at 0.250 and its
                        # fsq2 round-trip at 0.244; raw AMASS is already 1.205, and
                        # through vqvae 3.103.


def body_height(codes, device="cpu"):
    """Mean per-frame extent along the up axis, in metres."""
    from exemplars.nano_motion.render import features_to_gp
    from projects.nano_multimodal.decode.motion import codec
    feats = np.asarray(codec(device).decode(np.asarray(codes, dtype=np.int64)),
                       dtype=np.float64)
    if feats.ndim == 3:
        feats = feats[0]
    gp = features_to_gp(feats)
    return float((gp.max(axis=1) - gp.min(axis=1))[:, 2].mean())


def jitter(codes, device="cpu"):
    """Mean joint ACCELERATION in cm/frame^2, root-relative. This is what "the body
    shakes" is, numerically — and unlike MPJPE it is a time derivative, so a codec
    whose per-frame error is modest but uncorrelated frame to frame scores badly here
    and well on MPJPE. Choosing a codec on MPJPE alone picks the shaky one."""
    from exemplars.nano_motion.render import features_to_gp
    from projects.nano_multimodal.decode.motion import decode_features
    gp = features_to_gp(decode_features(codes, device))
    gp = gp - gp[:, 0:1]
    if len(gp) < 3:
        return float("nan")
    acc = gp[2:] - 2 * gp[1:-1] + gp[:-2]
    return float(np.linalg.norm(acc, axis=-1).mean() * 100)


def check_cache(path, key=None, n=8, device="cpu"):
    d = np.load(path, allow_pickle=True)
    key = key or ("codes" if "codes" in d.files else d.files[0])
    hs, js = [], []
    for i in range(min(n, len(d[key]))):
        c = np.asarray(d[key][i], dtype=np.int64).reshape(-1)
        if len(c) >= 4:
            hs.append(body_height(c, device))
            js.append(jitter(c, device))
    return float(np.mean(hs)), float(np.mean(js)), len(hs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", action="store_true",
                    help="probe every cache in outputs/motion_caches")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    if args.survey:
        print(f"{'cache':44s} {'height':>8s} {'jitter':>7s}   verdict")
        for f in sorted(glob.glob(str(spec.MOTION_CACHE_DIR / "*.npz"))):
            try:
                h, j, n = check_cache(f, n=args.n)
                print(f"{os.path.basename(f):44s} {h:6.2f} m {j:7.3f}   "
                      f"{'pairs' if h >= UPRIGHT_M else 'DOES NOT PAIR'}  ({n} clips)")
            except Exception as e:                       # noqa: BLE001 — a survey
                print(f"{os.path.basename(f):44s}      --   {type(e).__name__}")
        return

    config = assembly.load_config("motion")
    for split in ("train", "val"):
        key = "sources" if split == "train" else "val_sources"
        src = config["data"][key][0]
        facts = assembly.motion_cache_facts(src)
        path = spec.MOTION_CACHE_DIR / src["cache"]
        h, j, n = check_cache(path, n=args.n)
        tag = facts["source"]
        assert h >= UPRIGHT_M, (
            f"{src['cache']} decodes to a body {h:.2f} m tall through "
            f"{spec.MOTION_CODEC} — it does not pair with this codec, whatever its "
            f"tag says. See this module's docstring for the survey.")
        assert j <= SMOOTH, (
            f"{src['cache']} decodes to a body that shakes at {j:.3f} cm/frame^2 "
            f"through {spec.MOTION_CODEC} (a well-matched pair sits near 0.25, the "
            f"raw features' own level). The pose is upright but the MOTION is not "
            f"usable — which MPJPE would not have told you.")
        print(f"  {split:5s} {src['cache']:26s} codec={facts['codec']} ({tag}) "
              f"-> body {h:.2f} m, jitter {j:.3f} over {n} clips  ✓")
    print("\nOK — the configured motion caches decode to an upright human "
          "that does not shake.")


if __name__ == "__main__":
    main()
