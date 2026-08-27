"""
test_serve.py — every endpoint, on a real server, on all three lines.

The web is where the project is judged by looking, so the thing worth testing is
not the JSON shapes but the CLAIMS the panels make:

    * the browser shows the batches TRAINING sees — checked by asserting the row
      the API returns is the row the loader produces, id for id;
    * one component serves three lines — checked by requiring every line to answer
      every browse endpoint with the same keys;
    * the two panels share a renderer — checked by requiring a generated row and a
      dataset row to come back with the same fields.

Starts its own server on a free port so it never depends on one being up, and
never talks to one a human is using.

    python -m projects.nano_multimodal.tests.test_serve
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request

from projects.nano_multimodal import spec

BROWSE_KEYS = {"line", "row", "kind", "cells", "groups", "units", "notes",
                "n_supervised", "n_tokens", "media"}


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get(port, path, timeout=300, **q):
    url = f"http://127.0.0.1:{port}{path}"
    if q:
        url += "?" + urllib.parse.urlencode(q)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        d = json.loads(r.read())
    if isinstance(d, dict) and "error" in d:
        raise RuntimeError(f"{path}: {d['error'][:400]}")
    return d


def device_defaults():
    """The device the server picks must always carry an INDEX, and pinning it must
    work. Checked in-process and in milliseconds, because the failure it guards is a
    silent thread death whose only symptom is that every request hangs."""
    from projects.nano_multimodal.serve.app import _pin, pick_device
    d = pick_device()
    assert d == "cpu" or ":" in d, (
        f"pick_device() returned {d!r} with no device index — torch.cuda.set_device "
        f"rejects that, and the worker thread that calls it serves every request")
    _pin(d)
    import torch
    if torch.cuda.is_available():
        _pin("cuda")      # the un-indexed form must be normalised, not raise
    print(f"  device     pick_device() -> {d}; bare 'cuda' normalises")


def main():
    device_defaults()
    port = free_port()
    # NO --device: the server picks its own, and the DEFAULT is what students run.
    # Passing "--device cuda" here once hid a bug for a whole session — on a two-card
    # box the default is cuda:1, the model landed on card 0 anyway, and every
    # generation died on a device mismatch while every test passed.
    # ONE VISIBLE GPU, on purpose, and it is the whole reason this test exists in
    # this shape. That is what a student with one card has, and it is the
    # configuration that was broken: pick_device() returned a bare "cuda",
    # torch.cuda.set_device rejects that, the worker thread raised before its loop,
    # and no request was ever answered — every call hung until the client timed out.
    # The two-card default kept passing throughout, because there pick_device()
    # returns an indexed cuda:1. Running the server the way the SMALLER machine runs
    # it is what makes that class of bug fail here instead of at a student's desk.
    #
    # It cost two ten-minute timeouts and two wrong diagnoses before it was found:
    # first "the GPU is busy" (a re-run, without reading the traceback), then "an
    # unread pipe filled its 64KB buffer and blocked the child" — which is a real
    # failure mode, and was not this one: the server writes ~2.3KB per run. Writing
    # the child's output to a FILE is still right, and is kept; it just was never the
    # cause. The traceback said "the client is waiting" from the first run, and the
    # answer was in the SERVER's log, which nobody opened.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    # outputs/ holds everything this project WRITES and does not exist in a fresh
    # clone. Create it rather than crash on the first run someone makes.
    spec.OUTPUTS.mkdir(parents=True, exist_ok=True)
    log = open(spec.OUTPUTS / "serve-test.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "projects.nano_multimodal.serve.app",
         "--port", str(port), "--batch", "4"],
        cwd=spec.REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            try:
                get(port, "/api/lines", timeout=5); break
            except Exception:                                # noqa: BLE001
                time.sleep(1)
        else:
            raise SystemExit("server never came up")

        lines = get(port, "/api/lines")
        ready = {m["line"] for m in lines["models"] if m["ready"]}
        print(f"  lines      {[l['line'] for l in lines['lines'] if l['ok']]} | "
              f"checkpoints for {sorted(ready) or 'none'}")

        for line in ("text", "motion", "video"):
            v = get(port, "/api/vocab", line=line)
            b = get(port, "/api/rows", line=line)
            r = get(port, "/api/row", line=line, i=0)
            t = get(port, "/api/token", line=line, i=0, pos=1)
            s = get(port, "/api/stats", line=line)

            assert BROWSE_KEYS <= set(r), f"{line}: row is missing {BROWSE_KEYS - set(r)}"
            assert len(r["units"]) == r["n_tokens"], f"{line}: one unit per position"
            # Every cell carries BOTH halves of the screen's claim: the integer the
            # model sees and the readable piece it stands for.
            for c in r["cells"]:
                assert {"pos", "id", "band", "text", "sup"} <= set(c)
            assert sum(c.get("repeat", 1) for c in r["cells"]) == r["n_tokens"], (
                f"{line}: the cells do not account for every token — a collapsed run "
                f"is dropping or double-counting positions")
            assert v["vocab_size"] == sum(x["size"] for x in v["bands"])
            assert b["seq_len"] == r["n_tokens"]
            n_media = len(r["media"]) if isinstance(r["media"], list) else 1
            print(f"  {line:6s}     vocab {v['vocab_size']:>6} in {len(v['bands'])} bands | "
                  f"{len(b['rows'])} rows x {b['seq_len']} | {len(r['cells'])} cells "
                  f"({r['n_tokens']} tokens) | media {n_media} | "
                  f"sup {100*s['supervised']/s['total']:.0f}%")

        # --- the row the API serves IS the row the loader produces ------------
        from projects.nano_multimodal import assembly
        cfg = assembly.load_config("video")
        vocab = assembly.assemble_vocab("video", cfg)
        loader = assembly.build_loader("video", cfg, vocab, "val", "cuda", batch_size=4)
        ids = next(iter(loader))["idx"][0].cpu().numpy().tolist()
        cells = get(port, "/api/row", line="video", i=0)["cells"]
        served = [c["id"] for c in cells if not c.get("repeat")]
        assert served == ids[:len(served)], "the browser is not showing the loader's row"
        print("  identity   /api/row returns the loader's row id for id")

        # --- the inference panel, where a checkpoint exists -------------------
        # Generation is where the device bug lived: the browser builds no model, so it
        # was green throughout. Every line with a checkpoint must actually generate.
        if "text" in ready:
            t = get(port, "/api/gen/text", prompt="The city of", max_new=32)
            assert t["n_tokens"] > 0 and t["text"], "the text line generated nothing"
            print(f"  gen text   {t['n_tokens']} tokens: {t['text'][:56]!r}")
        if "motion" in ready:
            g = get(port, "/api/gen/motion", caption="a person walks forward.")
            assert {"spans", "units", "notes", "media"} <= set(g), \
                "a generated row does not carry the same fields as a dataset row"
            # SAME PICTURE, not just the same function. The two panels share
            # decode.render, but sharing a function does not share its arguments —
            # the inference panel once passed size=192 while the browser used 300,
            # and the same model looked worse on one tab than the other.
            import base64 as _b64, io as _io
            from PIL import Image as _Image
            def dims(u):
                return _Image.open(_io.BytesIO(_b64.b64decode(u.split(",", 1)[1]))).size
            ds = get(port, "/api/row", line="motion", i=0)["media"]
            assert dims(g["media"][0]) == dims(ds[0]), (
                f"the inference panel renders motion at {dims(g['media'][0])} and the "
                f"browser at {dims(ds[0])} — a student comparing them would be "
                f"comparing render sizes, not models")
            print(f"  gen motion {len(g['media'])} frames at {dims(g['media'][0])}, "
                  f"same renderer AND same size as a dataset row")
            p = get(port, "/api/gen/prompts")
            marked = sum(1 for x in p["prompts"] if x["in_training"])
            print(f"  prompts    {len(p['prompts'])} offered, {marked} marked as being "
                  f"IN the training set ({p['checked']} captions checked)")
        if "video" in ready:
            # The rollout is ENDLESS: past the row's capacity it re-anchors onto the
            # tail of the old window instead of stopping. Drive it past that point,
            # because "it raises after four frames" is exactly the bug this replaced.
            # HOW MANY STEPS IT TAKES. `room` is n_blocks - frames_done and does NOT
            # depend on carry — carry only narrows the room of the rows AFTER a
            # re-anchor (it restarts them at frames_done = carry - 1). The FIRST row
            # always takes n_blocks frames, so the earliest possible re-anchor is
            # step n_blocks + 1, whatever carry is. Asking for it sooner tests
            # nothing and fails on a healthy session.
            n = spec.video_shape()["n_blocks"]
            td = spec.video_shape()["td"]
            steps = n + 1
            s0 = get(port, "/api/session/new", carry=n)
            t0 = time.perf_counter()
            for i in range(steps):
                st = get(port, "/api/session/step", id=s0["id"], action=8)
                assert len(st["media"]) == td, "a step must return exactly one frame"
                assert st["frame"] == i + 1
            dt = (time.perf_counter() - t0) / steps
            assert st["reanchors"] >= 1, (
                "the session never re-anchored — it is still bounded by one row")
            print(f"  session    {st['frame']} frames, {st['reanchors']} re-anchors "
                  f"(carry={n}), {dt*1000:.0f} ms/frame")

        print("\nOK — every endpoint answers on every line, and the browser is "
              "reading the training loader.")
    finally:
        log.close()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
