"""
test_decode.py — the decoders, on real rows from the real loader.

A renderer that is subtly wrong poisons every judgement made by looking, and the
judgements in this project are all made by looking. So each line is checked on rows
the training loader actually produced, and each check is the one that would catch
the mistake that line is prone to:

    text    the per-token units must REASSEMBLE the decoded string exactly. That is
            the byte-straddling bug: decode each token separately and concatenate,
            and CJK text grows replacement characters at token boundaries while
            English looks fine.
    motion  codes -> features -> SMPL FK must produce 22 joints with a plausible
            standing height, and units must cover the frames 4:1.
    video   codes -> pixels must come back at the clip's resolution, and the unit
            grid must tile the frame exactly once: every 8x8 patch owned by exactly
            one integer, no gaps, no overlaps.

    python -m projects.nano_multimodal.tests.test_decode [--line text|motion|video]
"""

import argparse

import numpy as np
from projects.nano_multimodal import assembly, decode, spec


def one_row(line, device="cpu"):
    config = assembly.load_config(line)
    vocab = assembly.assemble_vocab(line, config)
    loader = assembly.build_loader(line, config, vocab, "val", device, batch_size=1)
    batch = next(iter(loader))
    return vocab, batch["idx"][0].cpu().numpy().tolist(), batch


def check_spans(row, vocab, r):
    assert r.spans[0]["end"] > 0
    assert r.spans[-1]["end"] == len(row), "spans must tile the row"
    for a, b in zip(r.spans, r.spans[1:]):
        assert a["end"] == b["start"], "spans must not gap or overlap"
    assert len(r.units) == len(row), "one unit per position"
    named = [s["label"] for s in r.spans if s["band"] == "control"]
    print(f"  spans      {len(r.spans)} runs; control labels: {named[:4]}")


def test_text():
    vocab, row, _ = one_row("text")
    r = decode.render("text", row, vocab)
    check_spans(row, vocab, r)
    # The units must REASSEMBLE the string. Concatenating per-token decodes would
    # not — that is the bug this asserts against.
    rebuilt = "".join(r.media[u["range"][0]:u["range"][1]] for u in r.units)
    assert rebuilt == r.media, "units do not reassemble the decoded text"
    nonempty = [u for u in r.units if u["range"][1] > u["range"][0]]
    print(f"  text       {len(r.media)} chars from {len(nonempty)} text tokens; "
          f"units reassemble exactly")
    print(f"  sample     {r.media[:70]!r}")
    print(f"  first 6    {[u['preview'] for u in r.units[:6]]}")


def test_motion_render_matches_reference():
    """The ported renderer must draw what the ORIGINAL draws.

    decode/motion.rasterise reuses its line artists across frames where the original
    (exemplars/nano_motion/render.gp_to_gif, itself the crystallization of the
    renderer behind the archived reference gallery) calls ax.cla() and rebuilds them.
    That is the only difference, and it is a performance change — so the pixels have
    to be the same. Checked here rather than asserted in a comment, because "I only
    changed how fast it is" is exactly the claim that turns out to be false.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from modalities.motion.data.converters import smpl_body as B
    from projects.nano_multimodal.decode import motion as M

    d = np.load(spec.MOTION_CACHE_DIR / "t2m_bones_val.npz", allow_pickle=True)
    gp = M.decode_joints(np.asarray(d["codes"][0], np.int64), "cpu")[:12]
    mine = M.rasterise(gp, size=280)

    # the ORIGINAL loop, verbatim: cla + rebuild every frame
    up, (h0, h1) = B.UP_AXIS, B.HORIZ_AXES
    mn, mx = gp.min(axis=(0, 1)), gp.max(axis=(0, 1))
    z_top, span, dpi = mx[up], 1.3, 100
    fig = plt.figure(figsize=(280 / dpi, 280 / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ref = []
    for t in range(len(gp)):
        ax.cla()
        P = gp[t]
        for a, b in M.SMPL_BONES:
            ax.plot([P[a, h0], P[b, h0]], [P[a, h1], P[b, h1]], [P[a, up], P[b, up]],
                    "-o", ms=2, lw=2, color="steelblue")
        ax.set_xlim(P[0, h0] - span, P[0, h0] + span)
        ax.set_ylim(P[0, h1] - span, P[0, h1] + span)
        ax.set_zlim(min(0, mn[up]), max(z_top, mn[up] + 2 * span))
        ax.set_box_aspect([1, 1, 1])
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        fig.canvas.draw()
        ref.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    ref = np.stack(ref)

    diff = np.abs(mine.astype(int) - ref.astype(int))
    bad = int((diff.max(axis=-1) > 8).sum())
    print(f"  render     {len(gp)} frames vs the rebuild-every-frame original: "
          f"max|Δ| = {diff.max()}, {bad} pixels differ by more than 8")
    assert bad == 0, "artist reuse changed the picture, not just the speed"


def test_motion():
    vocab, row, _ = one_row("motion")
    r = decode.render("motion", row, vocab, device="cpu", size=128)
    check_spans(row, vocab, r)
    mo = [u for u in r.units if u["band"] == "motion"]
    assert mo, "no motion tokens in the row"
    ds = spec.MOTION_DOWNSAMPLE
    assert mo[0]["frames"] == [0, ds] and mo[1]["frames"] == [ds, 2 * ds], \
        "motion units must cover frames 4:1"
    frames = r.media
    assert frames.shape[0] == len(mo) * ds, \
        f"{frames.shape[0]} frames for {len(mo)} codes (expected {len(mo)*ds})"
    gp = decode.motion.decode_joints([u["code"] for u in mo], device="cpu")
    assert gp.shape[1:] == (22, 3), f"FK returned {gp.shape}"
    height = float(gp[..., 2].max() - gp[..., 2].min())
    assert 0.8 < height < 2.6, f"implausible body height {height:.2f}m — check the FK"
    print(f"  motion     {len(mo)} codes -> {frames.shape[0]} frames "
          f"{frames.shape[1:]}, body {height:.2f}m")
    print(f"  notes      {r.notes}")


def test_video():
    vocab, row, _ = one_row("video")
    r = decode.render("video", row, vocab, device="cuda")
    check_spans(row, vocab, r)
    vids = [u for u in r.units if u["band"] == "video"]
    acts = [u for u in r.units if u["band"] == "action"]
    shape = spec.video_shape()
    assert len(vids) == shape["code_len"], f"{len(vids)} != {shape['code_len']}"
    assert len(acts) == shape["n_action"], f"{len(acts)} != {shape['n_action']}"

    # THE tiling check: within one latent frame, the units' 8x8 boxes must cover
    # every pixel exactly once. An off-by-one in the grid maths shows up here and
    # essentially nowhere else.
    res, ds = spec.RES, spec.CODEC_SPATIAL_DS
    cover = np.zeros((res, res), dtype=np.int32)
    for u in vids:
        if u["frame"] != 0:
            continue
        x, y, w, h = u["box"]
        cover[y:y + h, x:x + w] += 1
    assert cover.min() == 1 and cover.max() == 1, \
        f"patch tiling is wrong: coverage {cover.min()}..{cover.max()}"

    frames = r.media
    assert frames.shape[1:] == (res, res, 3), f"decoded {frames.shape}"
    assert frames.dtype == np.uint8
    print(f"  video      {len(vids)} codes -> {frames.shape[0]} frames "
          f"{frames.shape[1:]}; {ds}x{ds} patches tile {res}x{res} exactly once")
    print(f"  notes      {r.notes}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", choices=["text", "motion", "video"], action="append")
    args = ap.parse_args()
    for line in (args.line or ["text", "motion", "video"]):
        print(f"\n--- {line} ---")
        {"text": test_text, "motion": test_motion, "video": test_video}[line]()
        if line == "motion":
            test_motion_render_matches_reference()
    print("\nOK — the decoders agree with the rows the loader produces.")


if __name__ == "__main__":
    main()
