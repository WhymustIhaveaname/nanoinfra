"""playtest.py — 无头自测：照着网页那套输入逻辑打一局，把结果录成 mp4。

存在的理由：网页里的鼠标锁和 canvas 没法在服务器上点，但真正要验的不是 UI 事件，
而是「这串动作发过去，模型回的画面对不对」。所以这里直接打 server.py 的 HTTP 接口，
发一段有意图的动作序列（前进、左转、右转、开火、后退），把每帧存下来拼成视频，
顺带统计每个动作的耗时。
"""

import argparse
import base64
import io
import json
import http.client
import subprocess
import time

import numpy as np
from PIL import Image

ACTIONS = {"NOOP": 0, "TL": 1, "TR": 2, "BACK": 3, "TL+BACK": 4, "TR+BACK": 5,
           "MR": 6, "ML": 7, "FWD": 8, "TL+FWD": 9, "TR+FWD": 10, "ATK": 11}

# 一段有意图的操作，覆盖每一类动作，且刻意包含「转开再转回」——
# 世界模型最该表现出的性质就是转回去时房间还在原地。
SCRIPT = [("FWD", 12), ("TL", 10), ("FWD", 8), ("TR", 20), ("FWD", 8),
          ("TL", 10), ("ATK", 4), ("TL+FWD", 10), ("BACK", 6),
          ("MR", 6), ("ML", 6), ("NOOP", 4), ("TR+FWD", 10)]


# 复用一条 keep-alive 连接，和浏览器一样。每次新建连接会让服务端每帧都付
# 一次 CUDA per-thread 初始化（实测 6.2s vs 0.30s）——server.py 里已经用固定
# GPU 线程消掉了这笔开销，但客户端复用连接依然是对的做法。
_CONN = {}


def _conn(api):
    if "c" not in _CONN:
        host = api.split("//", 1)[-1]
        h, _, port = host.partition(":")
        _CONN["c"] = http.client.HTTPConnection(h, int(port or 80), timeout=300)
    return _CONN["c"]


def post(api, path, obj):
    c = _conn(api)
    c.request("POST", path, json.dumps(obj), {"Content-Type": "application/json"})
    return json.load(c.getresponse())


def get(api, path):
    c = _conn(api)
    c.request("GET", path)
    return json.load(c.getresponse())


def decode(data_url):
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:25677")
    ap.add_argument("--out", default="../outputs/gamengen/playtest.mp4")
    ap.add_argument("--fps", type=int, default=8, help="回放帧率（模型自身产不到这么快）")
    a = ap.parse_args()

    info = get(a.api, "/info")
    print(f"GPU {info['gpu']} | {info['steps']} 步去噪 | 上下文 {info['buffer']} 帧")
    if info.get("bench"):
        print(f"启动基准: {info['bench']}")

    s = post(a.api, "/new", {"seed": 3})
    frames = [decode(s["frame"])]
    sid, per_action, t_all = s["id"], {}, time.time()

    for name, n in SCRIPT:
        ts = []
        for _ in range(n):
            d = post(a.api, "/step", {"id": sid, "action": ACTIONS[name]})
            frames.append(decode(d["frame"]))
            ts.append(d["ms"])
        per_action[name] = round(float(np.mean(ts)), 1)
        print(f"  {name:8s} ×{n:3d}  {per_action[name]:6.1f} ms/帧", flush=True)

    wall = time.time() - t_all
    n = len(frames) - 1
    print(f"\n共 {n} 帧, 墙钟 {wall:.1f}s, 平均 {wall/n*1000:.0f} ms/帧 "
          f"= {n/wall:.2f} fps")

    w, h = frames[0].size
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.asarray(f, np.uint8).tobytes())
    p.stdin.close()
    assert p.wait() == 0
    print(f"-> {a.out}")

    # 一张横向长条，方便一眼扫过整局
    picks = [frames[i] for i in range(0, len(frames), max(1, len(frames) // 12))][:12]
    strip = Image.new("RGB", (w * len(picks), h))
    for i, f in enumerate(picks):
        strip.paste(f, (i * w, 0))
    strip.save(a.out.replace(".mp4", "_strip.png"))
    print(f"-> {a.out.replace('.mp4', '_strip.png')}")


if __name__ == "__main__":
    main()
