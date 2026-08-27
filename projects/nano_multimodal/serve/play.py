"""
serve/play.py — the inference panel's backend.

Three tabs, one sampler (inference.generate), one renderer (decode/) — the same
renderer the data browser uses, so any difference a student sees between the
training data and the model's output is the model.

    text     prompt -> continuation, with the same per-token colouring the browser
             puts on training text
    motion   a caption -> band-masked codes -> stick figure.
             The prompt list marks which captions appear in the TRAINING set:
             450,594 template-generated captions is few enough that memorisation
             is on the table, so a training caption that generates beautifully
             proves less than it looks like it does. Making that visible is the
             lesson; hiding it teaches the opposite.
    video    a seed observation + the student's keypresses -> a rollout

VIDEO IS PULL, AND THAT REMOVES A WHOLE LAYER. The research line's server owns a
game clock, pushes a block every 114 ms, and needs a WebSocket with ack-based
backpressure to keep from running ahead of the client. At ~250 ms per AR frame
there is no clock to run ahead of: the client asks for one frame and waits for it.
One ordinary HTTP request per frame, no socket, no backpressure, no reconnect
logic. The honest interaction is also the simpler one.

Sessions are held by id. A session owns a KV cache — several hundred MB — so they
are capped and the oldest is dropped rather than accumulating until the box dies.
"""

import itertools

from projects.nano_multimodal import decode, inference, spec

_MODELS = {}
_SESSIONS = {}
_IDS = itertools.count(1)
MAX_SESSIONS = 4

DEVICE = ["cuda"]


def list_checkpoints(line):
    """Every loadable checkpoint for a line, best first, each labelled by what it IS.

    The labels are the point. On an overfitting line the same run holds its own
    counter-example: a floor and, thousands of steps later, a visibly worse model
    trained on the same data with the same recipe. Letting a student switch between
    those two and look is a better argument than any curve, so the list says which is
    which rather than leaving it to be read off step numbers.
    """
    from pathlib import Path
    from collections import defaultdict

    by_run = defaultdict(list)
    for p in inference.checkpoints(line):
        by_run[Path(p).parent].append(Path(p))
    out, best_val = [], None
    for run, paths in by_run.items():
        val = inference._val_by_step(run / "metrics.jsonl")
        steps = sorted(paths, key=inference._step_of)
        floor = min(steps, key=lambda p: val.get(inference._step_of(p), float("inf")))
        # THESE THREE TAG STRINGS ARE AN API. web/index.html matches "末尾" literally
        # to decide whether to show the overfitting warning beside the picker. Nothing
        # binds the two files together, so re-wording a tag here — a translation, a
        # tidier phrase, a stray space — removes that warning silently: no error, no
        # console message, just a panel that stops teaching the lesson it exists for.
        # Change one end and grep the other.
        for p in steps:
            v = val.get(inference._step_of(p))
            tags = []
            if p == floor:
                tags.append("谷底")
            if p == steps[-1] and len(steps) > 1 and p != floor:
                tags.append("末尾")
            out.append({"path": str(p), "run": run.name, "step": inference._step_of(p),
                        "val": v, "tags": tags})
    out.sort(key=lambda c: (c["val"] if c["val"] is not None else float("inf")))
    if out:
        out[0]["tags"] = ["★ 全局最好"] + out[0]["tags"]
    return {"line": line, "checkpoints": out}


def available():
    """Which lines have a checkpoint, newest first."""
    out = []
    for line in ("text", "motion", "video"):
        cks = inference.checkpoints(line)
        out.append({"line": line, "checkpoints": [str(c) for c in cks],
                    "ready": bool(cks)})
    return {"models": out}


def resolve(line, ckpt=None):
    """Which checkpoint a request will actually use, and what its val CE was. The
    panel shows this: a student who has trained three times needs to know which of
    their models just spoke."""
    from pathlib import Path
    if ckpt is None:
        found = inference.checkpoints(line)
        if not found:
            return None
        ckpt = found[0]
    p = Path(ckpt)
    val = inference._val_by_step(p.parent / "metrics.jsonl").get(inference._step_of(p))
    return {"path": str(p), "run": p.parent.name, "step": inference._step_of(p),
            "val": val}


def model(line, ckpt=None):
    """Load once and keep. Three models plus three decoders is a real residency —
    which is why app.py defaults --device to the second GPU."""
    key = (line, ckpt)
    if key not in _MODELS:
        _MODELS[key] = inference.load(line, ckpt, device=DEVICE[0])
    return _MODELS[key]


def sample_text(prompt, ckpt=None, max_new=120, temperature=0.9, top_k=40, seed=0):
    system, vocab = model("text", ckpt)
    used = resolve("text", ckpt)
    ids, text = inference.sample_text(system, vocab, prompt, max_new=max_new,
                                      temperature=temperature, top_k=top_k, seed=seed,
                                      device=DEVICE[0])
    r = decode.render("text", ids, vocab)
    return {"text": text, "units": r.units, "n_tokens": len(ids), "prompt": prompt,
            "checkpoint": used}


def sample_motion(caption, ckpt=None, temperature=0.9, top_k=40, seed=0):
    system, vocab = model("motion", ckpt)
    used = resolve("motion", ckpt)
    row = inference.sample_motion(system, vocab, caption, temperature=temperature,
                                  top_k=top_k, seed=seed, device=DEVICE[0])
    # NO size argument. Both panels take it from spec.RENDER_SIZE, so a student
    # comparing a dataset row against a generated one is comparing MODELS.
    r = decode.render("motion", row, vocab, device=DEVICE[0])
    return {"kind": "motion", "spans": r.spans, "units": r.units, "notes": r.notes,
            "media": r.media, "caption": caption, "checkpoint": used}


# Captions offered by the panel. Short ones on purpose: a long caption describes a
# compound action ("...then turns, then raises their arms, then..."), and with a model
# this size the second half is usually lost — which reads as "the model is bad" when
# what happened is that the prompt asked for four things.
MIN_CHARS, MAX_CHARS, N_FROM_TRAIN = 22, 62, 24


def motion_prompts(n=N_FROM_TRAIN):
    """The offered captions, each marked with whether it is IN the training set.

    A student who picks a training caption and sees a beautiful result has watched
    MEMORISATION, not generalisation. This line has ~450k (caption, motion) pairs
    against a model that will happily memorise them, so the distinction is the whole
    lesson — and it only lands if both sides are on screen at once. Hence: the
    hand-picked prompts from spec (which are NOT in this corpus, and are the ones the
    archived gallery used), plus a spread of captions taken VERBATIM from the training
    set.

    Deterministic: sorted by length, then evenly sampled. Two students see the same
    list, and the list does not shuffle under them between page loads.
    """
    import numpy as np

    from projects.nano_multimodal import assembly

    train = []
    try:
        config = assembly.load_config("motion")
        d = np.load(spec.MOTION_CACHE_DIR / config["data"]["sources"][0]["cache"],
                    allow_pickle=True)
        seen = set()
        for caps in d["captions"][:20000]:
            for c in (caps if isinstance(caps, (list, np.ndarray)) else [caps]):
                c = " ".join(str(c).split())
                if MIN_CHARS <= len(c) <= MAX_CHARS and c.lower() not in seen:
                    seen.add(c.lower()); train.append(c)
    except Exception:                                        # noqa: BLE001
        pass

    out = [{"caption": p, "in_training": False} for p in spec.MOTION_PROMPTS]
    if train:
        train.sort(key=lambda c: (len(c), c.lower()))
        step = max(1, len(train) // n)
        out += [{"caption": c, "in_training": True} for c in train[::step][:n]]
    return {"prompts": out, "checked": len(train)}


def session_new(ckpt=None, seed=0, temperature=1.0, top_k=40, row=None, carry=2):
    """Start a rollout from a REAL observation out of the val split — the model was
    trained to continue an observation, not to invent one from nothing."""
    from projects.nano_multimodal import assembly

    system, vocab = model("video", ckpt)
    config = assembly.load_config("video")
    loader = assembly.build_loader("video", config, vocab, "val", DEVICE[0], batch_size=8)
    b = next(iter(loader))
    idx = b["idx"].cpu().numpy()
    i = 0 if row is None else int(row) % idx.shape[0]
    shape = spec.video_shape()
    v_off = vocab.layout.offset(spec.VIDEO_TYPE_ID)
    seed_codes = [int(t) - v_off for t in idx[i][2:2 + shape["codes_per_frame"]]]

    sid = str(next(_IDS))
    while len(_SESSIONS) >= MAX_SESSIONS:                    # a KV cache is not small
        _SESSIONS.pop(next(iter(_SESSIONS)))
    _SESSIONS[sid] = {
        "session": inference.Session(system, vocab, seed_codes, seed=seed,
                                     temperature=temperature, top_k=top_k,
                                     device=DEVICE[0], sampler="static", carry=carry),
        "vocab": vocab,
    }
    from projects.nano_multimodal.decode.video import decode_pixels
    px = decode_pixels(seed_codes, device=DEVICE[0])
    return {"id": sid, "endless": True, "media": (px * 255).astype("uint8"),
            "checkpoint": resolve("video", ckpt),
            "actions": [{"id": i, "name": n} for i, n in enumerate(spec.ACTION_NAMES)]}


def session_step(sid, action):
    """One button -> one latent frame. Roughly 250 ms; the client waits for it.

    Returns only the NEW pixel frames. The rollout is endless, so returning the whole
    history would grow without bound — and decoding it would too.

    What IS decoded is the current WINDOW (at most `n_latent` latent frames), not the
    new frame alone: the codec is temporal, and a latent frame decoded by itself is
    not the same pixels as that frame decoded inside its clip. Decoding the window and
    keeping its tail costs one bounded decode per step and gets the context right.
    """
    from projects.nano_multimodal.decode.video import decode_pixels
    s = _SESSIONS.get(str(sid))
    if s is None:
        return {"error": "session expired — start a new rollout"}
    sess = s["session"]
    codes = sess.step(int(action))
    td = sess.shape["td"]
    flat = [c for frame in sess.window for c in frame]
    px = decode_pixels(flat, device=DEVICE[0])[-td:]
    return {"id": str(sid), "codes": codes,
            "frame": sess.total_frames, "reanchors": sess.reanchors,
            "room": sess.room,
            "action": spec.ACTION_NAMES[int(action) % spec.N_ACTIONS],
            "media": (px * 255).astype("uint8")}
