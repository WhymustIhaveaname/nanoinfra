"""
serve/app.py — one process, one port, two panels.

    python -m projects.nano_multimodal.serve.app --port 8800 --device cuda:1

TRANSPORT ONLY. This file owns the HTTP plumbing, static files and the single GPU
worker thread; it owns no knowledge of any modality. The two panels live next door
and are pure logic:

    browse.py   what the training data looks like  (loader + decode, no model)
    play.py     what the model produced            (model + inference + decode)

Both import decode/; neither imports the other.

WHY ONE PROCESS AND NOT TWO. The two panels decode the same rows with the same
code — the browser shows a row from the dataset, the inference panel shows a row
the model just wrote, and a student is meant to compare them. Two servers would
mean two copies of the decoders, two GPU residencies (the Cosmos decoder alone is
~86MB of TorchScript), two ports and two commands. One process, one URL, and the
comparison is a tab switch.

ONE GPU WORKER THREAD. ThreadingHTTPServer answers requests concurrently, but the
decoders and the models are not thread-safe and the GPU is one device. Every
handler that touches the GPU is run on a single worker over a queue — the shape the
research line's servers converged on independently.

NO WEBSOCKET, and that is a consequence of being autoregressive rather than a
shortcut. The research line pushes a diffusion block every 114 ms and needs a
socket with ack-based backpressure to avoid running ahead of the client. At ~250 ms
per AR frame the client simply asks for the next frame and waits: one ordinary
request per frame, no socket, no reconnect logic. The honest interaction is the
simpler one to build.

--device DEFAULTS TO THE SECOND GPU when there is one. The web holds three models
plus three decoders resident; a student running it on the card they are training on
will OOM, and the error will not say why.

NO BUILD STEP. stdlib http.server + PIL, one self-contained index.html. A teaching
project must not ask a student to install a frontend toolchain before they can look
at their own data.
"""

import argparse
import base64
import io
import json
import queue
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
_JOBS = queue.Queue()


# --------------------------------------------------------------------------
# the single GPU worker
# --------------------------------------------------------------------------

def _worker(device=None):
    # PIN THE DEVICE HERE, inside the worker. torch.cuda.set_device is PER-THREAD, so
    # setting it in main() leaves this thread — the only one that ever touches the GPU
    # — still defaulting to card 0. core's build_system takes torch.device("cuda") on
    # a single-GPU launch, so the model landed on card 0 while every input tensor was
    # built on the requested card, and each generate() died with "index is on cuda:1,
    # different from other tensors on cuda:0".
    #
    # The data browser was unaffected (it builds no model), which is why this sat in
    # the DEFAULT configuration — pick_device() returns cuda:1 on a two-card box —
    # while every test that passed had been run with an explicit --device cuda.
    if device and str(device).startswith("cuda"):
        import torch
        torch.cuda.set_device(torch.device(device))
    while True:
        fn, args, kw, out = _JOBS.get()
        try:
            out.put(("ok", fn(*args, **kw)))
        except Exception as e:                               # noqa: BLE001
            out.put(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def gpu(fn, *args, **kw):
    """Run fn on the GPU worker and wait. Every handler goes through here, so the
    device is touched by exactly one thread no matter how many requests arrive."""
    out = queue.Queue()
    _JOBS.put((fn, args, kw, out))
    kind, val = out.get()
    if kind == "err":
        raise RuntimeError(val)
    return val


# --------------------------------------------------------------------------
# media encoding
# --------------------------------------------------------------------------

def _data_url(img):
    """LOSSLESS WebP. These frames are 128px and get drawn at 3x, so a lossy encoder's
    ringing would be indistinguishable from the codec's own reconstruction blur — and
    telling those apart is the entire point of looking at a decoded frame."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="WEBP", lossless=True)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def encode_media(payload):
    """Turn a Rendered.media into something JSON can carry.

    Frames become one data URL each rather than a single animated file, because the
    browser scrubs them against the token track: it needs to show frame k when the
    user hovers token k, which an animated GIF cannot do.
    """
    if payload is None or isinstance(payload, str):
        return payload
    arr = np.asarray(payload)
    if arr.ndim == 4:
        return [_data_url(f) for f in arr]
    if arr.ndim == 3:
        return [_data_url(arr)]
    return None


def jsonable(obj):
    if isinstance(obj, dict):
        return {k: (encode_media(v) if k == "media" else jsonable(v))
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------

def route(path, q):
    from projects.nano_multimodal.serve import browse, play

    one = lambda k, d=None: q.get(k, [d])[0]                 # noqa: E731

    if path == "/api/lines":
        return {**browse.lines(), **play.available()}
    if path == "/api/vocab":
        return gpu(browse.vocab, one("line", "text"))
    if path == "/api/rows":
        return gpu(browse.rows, one("line", "text"), one("split", "val"))
    if path == "/api/row":
        return gpu(browse.row, one("line", "text"), int(one("i", 0)),
                   one("split", "val"))
    if path == "/api/token":
        return gpu(browse.token, one("line", "text"), int(one("pos", 0)),
                   int(one("i", 0)), one("split", "val"))
    if path == "/api/stats":
        return gpu(browse.stats, one("line", "text"), one("split", "val"))

    if path == "/api/gen/text":
        return gpu(play.sample_text, one("prompt", "The city of"), one("ckpt"),
                   int(one("max_new", 120)), float(one("temperature", 0.9)),
                   int(one("top_k", 40)), int(one("seed", 0)))
    if path == "/api/gen/motion":
        return gpu(play.sample_motion, one("caption", "a person walks forward."),
                   one("ckpt"), float(one("temperature", 0.9)),
                   int(one("top_k", 40)), int(one("seed", 0)))
    if path == "/api/checkpoints":
        return gpu(play.list_checkpoints, one("line", "motion"))
    if path == "/api/gen/prompts":
        return gpu(play.motion_prompts)
    if path == "/api/session/new":
        return gpu(play.session_new, one("ckpt"), int(one("seed", 0)),
                   float(one("temperature", 1.0)), int(one("top_k", 40)),
                   one("row"), int(one("carry", 2)))
    if path == "/api/session/step":
        return gpu(play.session_step, one("id"), int(one("action", 8)))
    return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass                                                 # the console is for training

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, (HERE / "web" / "index.html").read_bytes(),
                              "text/html; charset=utf-8")
        if not u.path.startswith("/api/"):
            return self._send(404, "not found", "text/plain")
        try:
            result = route(u.path, parse_qs(u.query))
            if result is None:
                return self._send(404, json.dumps({"error": "no such endpoint"}))
            return self._send(200, json.dumps(jsonable(result)))
        except Exception as e:                               # noqa: BLE001
            return self._send(500, json.dumps({"error": str(e)}))

    do_POST = do_GET


def pick_device():
    """Default to the SECOND GPU when there is one. This process holds three models
    and three decoders; sharing a card with a training run OOMs, and the error does
    not say why."""
    try:
        import torch
        n = torch.cuda.device_count()
        return "cuda:1" if n > 1 else ("cuda" if n else "cpu")
    except Exception:                                        # noqa: BLE001
        return "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=8, help="rows shown in the batch view")
    args = ap.parse_args()

    from projects.nano_multimodal.serve import browse, play
    dev = args.device or pick_device()

    browse.DEVICE[0] = play.DEVICE[0] = dev
    browse.BATCH[0] = args.batch

    threading.Thread(target=_worker, args=(dev,), daemon=True).start()
    print(f"nano_multimodal — data browser + inference, on {dev}")
    print(f"  http://localhost:{args.port}")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
