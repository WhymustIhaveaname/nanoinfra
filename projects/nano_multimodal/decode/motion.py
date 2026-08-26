"""
decode/motion.py — codes -> a stick figure in a 3D box.

    global ids -> local codes -> codec.decode -> rot139 features [T,139]
                -> SMPL forward kinematics -> joint positions [T,22,3]
                -> a 3D box with a ground grid, and the skeleton inside it

WHY STICKS AND NOT A MESH. An earlier version skinned the full SMPL body (6890
vertices, LBS, shaded splat) and it was rejected by eye: a solid body reads as a
blob at 224px, its silhouette hides which way a limb is pointing, and judging "is
that pose right" needs to see the JOINTS. The archived reference gallery this
project is measured against draws sticks for the same reason. The mesh code is gone
rather than kept behind a flag: an option nobody picks is a maintenance cost with
no reader.

HOW. rasterise() is the reference renderer, ported: same matplotlib 3D axes, same
limits, same markers, same follow rule. The only change is that it updates its line
artists instead of rebuilding them every frame, which halves the cost for identical
output. A hand-rolled projection was tried first and rejected — 40x faster, and it
did not look the same.

TWO THINGS THE RENDERER MUST NOT HIDE:

  * Root translation. A camera that re-centres on the body every frame makes a model
    that slides across the floor look fine. This one FOLLOWS the root — otherwise a
    person who walks anywhere shrinks to a speck — but draws the BOX AND ITS GRID in
    world coordinates, so travel shows as the floor sliding underneath and skating
    feet stay visible. That is the reference gallery's trick, and it is why following
    is safe. It follows the root JOINT, not a body point: a point on the pelvis
    SURFACE moves when the hips rotate, and a camera tied to one slides sideways
    every time the body twists, taking the world with it.
  * The 4:1 ratio. One motion token is FOUR frames at 30fps (codec downsample 4), so
    every unit maps to its four frames and the browser can scrub token by token.
    "One integer = an eighth of a second of movement" is the fact this line exists to
    teach; it should be visible, not stated.

Needs models/smplh/ for the rest skeleton (22 joints + the kinematic tree). Without
it render() raises with that message and the text and video lines are unaffected,
which is why the import is lazy.
"""

import numpy as np

from projects.nano_multimodal import spec
from projects.nano_multimodal.decode import Rendered, band_of, spans_of

_CODEC = {}


def codec(device="cuda"):
    """The motion codec named by spec.MOTION_CODEC, from the shelf, loaded once."""
    import importlib
    if device not in _CODEC:
        mod = importlib.import_module(f"modalities.motion.tokenizers.{spec.MOTION_CODEC}")
        _CODEC[device] = mod.load(device=device)
    return _CODEC[device]


def decode_features(local_codes, device="cpu"):
    """local codes -> rot139 features [T, 139]."""
    feats = np.asarray(codec(device).decode(np.asarray(local_codes, dtype=np.int64)),
                       dtype=np.float64)
    return feats[0] if feats.ndim == 3 else feats


def decode_joints(local_codes, device="cpu"):
    """local codes -> global joint positions [T, 22, 3]. Kept for the tests, which
    check body height against the SKELETON rather than the mesh."""
    from exemplars.nano_motion.render import features_to_gp
    return features_to_gp(decode_features(local_codes, device))


def render(row, vocab, device="cpu", size=None, **_):
    ids = np.asarray(row, dtype=np.int64)
    tok = vocab.tokenizers.get("text")

    codes, text_ids, units = [], [], []
    for pos, tid in enumerate(ids):
        band, _type_id, local = band_of(vocab, int(tid))
        if band == "motion":
            k = len(codes)
            ds = spec.MOTION_DOWNSAMPLE
            units.append({"tokens": [pos, pos + 1], "band": "motion", "code": local,
                          "frames": [k * ds, (k + 1) * ds],
                          "preview": f"frames {k*ds}-{(k+1)*ds-1} ({ds/30:.2f}s)"})
            codes.append(local)
        elif band == "text":
            text_ids.append(int(tid))
            units.append({"tokens": [pos, pos + 1], "band": "text",
                          "preview": tok.decode([int(tid)]) if tok else ""})
        else:
            units.append({"tokens": [pos, pos + 1], "band": band,
                          "preview": vocab.resolver.name_of(int(tid)) or ""})

    caption = tok.decode(text_ids) if (tok and text_ids) else ""
    size = size or spec.RENDER_SIZE
    frames = (rasterise(decode_joints(codes, device), size=size) if codes
              else np.zeros((0, size, size, 3), dtype=np.uint8))
    return Rendered(
        kind="motion",
        spans=spans_of(row, vocab),
        units=units,
        media=frames,
        notes=[f"{len(codes)} codes = {len(codes)*spec.MOTION_DOWNSAMPLE} frames "
               f"({len(codes)*spec.MOTION_DOWNSAMPLE/30:.1f}s at 30fps); "
               f"one code = {spec.MOTION_DOWNSAMPLE} frames",
               f"caption: {caption}" if caption else "unconditional"],
    )


SMPL_BONES = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9),
              (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17),
              (16, 18), (17, 19), (18, 20), (19, 21)]


def rasterise(gp, size=280, max_frames=None, follow=True, span=1.3, title=""):
    """joint positions [T,22,3] -> [T,size,size,3] uint8.

    PORTED, not reinvented. Every drawing decision below is copied from
    exemplars/nano_motion/render.gp_to_gif, which is itself the verbatim
    crystallization of the renderer that made the archived reference gallery.
    Same axes, same limits, same `-o` markers at ms=2 / lw=2 / steelblue, same box
    aspect, same follow rule. matplotlib draws it, so the picture IS the gallery's
    picture rather than an imitation of it.

    ONE THING IS DIFFERENT, and only one: the original calls `ax.cla()` every frame
    and rebuilds all 21 line artists. Reusing them and updating their data halves the
    cost (29 -> 13 ms a frame at this size) for identical output, which is the
    difference between "fine for an offline gallery" and "usable in a web request".
    tests/test_decode.py checks the two against each other pixel for pixel.

    A hand-rolled projection was tried first and rejected: 40x faster, and it did not
    look the same. Looking the same was the requirement.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from modalities.motion.data.converters import smpl_body as B

    gp = np.asarray(gp, dtype=np.float64)
    if max_frames and len(gp) > max_frames:
        # The original caps long clips at 160 frames to keep a GIF small. Here the
        # cap is OFF by default, because the browser maps every motion token to its
        # four frames — "one integer = an eighth of a second" is the point of that
        # panel, and subsampling silently breaks the correspondence. Offline
        # galleries, which do write GIFs, pass the cap explicitly.
        gp = gp[np.linspace(0, len(gp) - 1, max_frames).astype(int)]
    up = B.UP_AXIS
    h0, h1 = B.HORIZ_AXES
    mn, mx = gp.min(axis=(0, 1)), gp.max(axis=(0, 1))
    ctr = (mn + mx) / 2
    fixed_span = max((mx - mn).max(), 1.0) / 2 * 1.1
    z_top = (mx[up] if follow else ctr[up] + fixed_span)

    dpi = 100
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    lines, out = None, []
    for t in range(len(gp)):
        P = gp[t]
        if lines is None:
            lines = [ax.plot([P[a, h0], P[b, h0]], [P[a, h1], P[b, h1]],
                             [P[a, up], P[b, up]], "-o", ms=2, lw=2,
                             color="steelblue")[0] for a, b in SMPL_BONES]
            ax.set_box_aspect([1, 1, 1])
            if title:
                ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("up")
            else:                       # the web already shows the caption above
                ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        else:
            for ln, (a, b) in zip(lines, SMPL_BONES):
                ln.set_data([P[a, h0], P[b, h0]], [P[a, h1], P[b, h1]])
                ln.set_3d_properties([P[a, up], P[b, up]])
        if follow:                      # centre on the root (joint 0) horizontally
            cx, cy = P[0, h0], P[0, h1]
            ax.set_xlim(cx - span, cx + span); ax.set_ylim(cy - span, cy + span)
            ax.set_zlim(min(0, mn[up]), max(z_top, mn[up] + 2 * span))
        else:
            ax.set_xlim(ctr[h0] - fixed_span, ctr[h0] + fixed_span)
            ax.set_ylim(ctr[h1] - fixed_span, ctr[h1] + fixed_span)
            ax.set_zlim(min(ctr[up] - fixed_span, 0), ctr[up] + fixed_span)
        if title:
            ax.set_title(f"{title}\nframe {t+1}/{len(gp)}", fontsize=9)
        fig.canvas.draw()
        out.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return np.stack(out)
