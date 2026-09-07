"""server.py — 把 GameNGen 复现模型包成一个 HTTP 服务，供预览页在线试玩。

为什么不直接用上游的 run_playable_env.py：那个脚本用 `keyboard` 模块直接读本机键盘
（Linux 上要 root），画面走 OpenCV 窗口，绑死在有显示器的本地机器上。这里改成
「模型常驻显存 + HTTP 收动作、回 PNG」，浏览器负责输入和显示，服务器不需要图形界面。

模型：三份社区复现权重，见 models.json；页面上可切换，按需加载（同时只驻留一份，
16GB 卡上每份约 4GB，留两份就没余量给推理的中间张量了）。Google 的官方权重从未公开。
推理代码取自 Masao-Taketani 的复现（锚点 692b5b32），原样放在 upstream/；
arnaudstiegler 的权重文件结构与之相同，所以同一套代码都能加载。

状态：每个 session 持有一个 64 帧的滚动上下文（context_latents）和同长的动作序列。
走一步 = 用这两者预测下一帧的 latent，解码成像素，然后把新 latent 推进窗口。
上下文只进不出，所以误差会累积——这是自回归世界模型的固有行为，不是 bug。
"""

import argparse
import base64
import io
import json
import random
import sys
import itertools
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "upstream"))

from diffusers.image_processor import VaeImageProcessor            # noqa: E402
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution  # noqa: E402
from PIL import Image                                              # noqa: E402

import config_sd                                                   # noqa: E402
import model as upstream_model                                     # noqa: E402
import run_inference                                               # noqa: E402
from config_sd import CFG_GUIDANCE_SCALE                           # noqa: E402
from model import load_model                                       # noqa: E402
from run_inference import (decode_and_postprocess, next_latent,    # noqa: E402
                           prepare_conditioning_frames)


def set_buffer_size(n):
    """上下文帧数是每个模型自己的事，但上游把它写成了模块级常量 BUFFER_SIZE=64。

    两个复现训的长度不一样——Masao 用 64 帧，arnaudstiegler 用 9 帧（从 unet 的
    in_channels 反推：4*(n+1)，260 -> 64，40 -> 9）。不改的话换模型必报
    「expected input to have 40 channels, but got 260」。
    run_inference 是 `from config_sd import BUFFER_SIZE`，名字已经绑进它自己的
    命名空间，所以三个模块都要改。同一时刻只有一个模型在显存里，全局改是安全的。
    """
    for m in (config_sd, run_inference, upstream_model):
        m.BUFFER_SIZE = n

# 上游 run_playable_env.select_action 的动作表，12 个
ACTIONS = ["NOOP", "TL", "TR", "BACK", "TL+BACK", "TR+BACK",
           "MR", "ML", "FWD", "TL+FWD", "TR+FWD", "ATK"]

_STATE = {}                     # sid -> session
_MODELS = {}                    # id -> 模型清单里的一条
_CUR = {"id": None, "engine": None, "bench": None, "load_ms": None}
_IDS = itertools.count(1)       # session id：必须单调，用 len(_STATE) 会重号

# 载入进度。/status 不排队进 GPU 线程，所以 GPU 忙着的时候照样读得到——
# 这才使得前端能在等 /load 返回的同时轮询出真实进度，而不是画一根假的滚动条。
_PROG = {"phase": "", "done": 0, "total": 0}


def progress(phase, done=0, total=0):
    _PROG.update(phase=phase, done=done, total=total)

# 所有 GPU 计算都排到这一个固定线程上执行。
#
# 这不是为了「串行化」（那用一把锁就够了），而是因为 CUDA 的 per-thread 初始化很贵：
# ThreadingHTTPServer 每条 TCP 连接开一个新线程，而一个全新线程第一次跑这个模型要
# 约 6 秒，之后才回落到 0.3 秒。浏览器靠 keep-alive 复用连接所以感觉不到，但任何
# 每次新建连接的客户端（urllib、curl 逐次调用）就会每帧都付这 6 秒——实测 6.2s vs
# 0.30s，20 倍。把计算钉在一个长期存活的线程上，这笔开销一辈子只付一次。
_JOBS = queue.Queue()


def _worker():
    while True:
        fn, done, box = _JOBS.get()
        try:
            box.append(("ok", fn()))
        except BaseException as e:                      # 别让工作线程死掉
            box.append(("err", e))
        finally:
            done.set()


def run_on_gpu(fn, timeout=180):
    """把 fn 丢给 GPU 线程执行并等结果。超时抛 TimeoutError，不静默挂死。"""
    done, box = threading.Event(), []
    _JOBS.put((fn, done, box))
    if not done.wait(timeout):
        raise TimeoutError("GPU 线程超时")
    kind, val = box[0]
    if kind == "err":
        raise val
    return val


class Engine:
    def __init__(self, unet_dir, vae_dir, latent_pt, device, steps, noise_level):
        self.device = torch.device(device)
        self.steps = steps
        self.noise_level = noise_level
        # 上游的 next_latent 用 autocast(float32) 跑，所以只能在 fp32 内部提速：
        # TF32 让 Ampere/Ada 的张量核吃下 fp32 的卷积和矩阵乘，尾数从 23 位降到 10 位。
        # 对扩散推理这点精度无所谓，但快得多。cudnn.benchmark 是因为形状每帧固定不变。
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        cfg = json.loads((Path(unet_dir) / "unet" / "config.json").read_text())
        self.buffer = cfg["in_channels"] // 4 - 1
        set_buffer_size(self.buffer)
        t0 = time.time()
        out = load_model(str(unet_dir), str(vae_dir), device=self.device)
        self.unet, self.vae, self.action_embedding, self.noise_scheduler = out[:4]
        self.unet.eval(); self.vae.eval()
        self.proc = VaeImageProcessor(
            vae_scale_factor=2 ** (len(self.vae.config.block_out_channels) - 1))
        # weights_only=True：这个 .pt 是从 HuggingFace 下的第三方文件，里面只有张量，
        # 没必要开着 pickle 的任意代码执行口子。
        self.episode = torch.load(latent_pt, map_location="cpu", weights_only=True)
        self.n_latent = len(self.episode["actions"])
        self.gpu_name = (torch.cuda.get_device_name(self.device)
                         if self.device.type == "cuda" else "cpu")
        print(f"[engine] 模型载入 {time.time()-t0:.1f}s, device={self.device}, "
              f"上下文 {self.buffer} 帧, 种子 episode {self.n_latent} 帧", flush=True)

    def _decode(self, latents):
        with torch.inference_mode():
            img = decode_and_postprocess(vae=self.vae, image_processor=self.proc,
                                         latents=latents, output_type="pt")
        return img

    def new_session(self, seed=None):
        """从种子 episode 里随机取 64 帧真实 latent 当开局上下文。"""
        if seed is not None:
            random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
        start = random.randint(0, self.n_latent - self.buffer - 1)
        params = self.episode["parameters"][start:start + self.buffer]
        latents = DiagonalGaussianDistribution(params).sample().to(self.device)
        # 只解最后一帧当开局画面。上游 run_playable_env 把 64 帧一起解，在 16GB 卡上
        # 一次要 5GB 直接 OOM——而那 63 帧解出来根本没人看。
        init = self._decode(latents[-1:] * self.vae.config.scaling_factor)
        ctx = prepare_conditioning_frames(self.vae, latents=latents,
                                          device=self.device, dtype=latents.dtype)
        acts = self.episode["actions"][start:start + self.buffer].to(self.device)
        s = {"ctx": ctx, "acts": acts, "steps": 0, "png": to_png(init)}
        # 空跑一帧把一次性成本吃掉。实测第一次 next_latent 要 4.9 秒而之后只要 0.3 秒
        # ——cuDNN 在新形状上自动调优 + GPU 升频。不预热的话玩家按下第一个键会卡 5 秒。
        # next_latent 不改动入参，所以结果丢掉即可，session 状态不受影响。
        next_latent(
            unet=self.unet, noise_scheduler=self.noise_scheduler,
            action_embedding=self.action_embedding,
            context_latents=ctx.unsqueeze(0), device=self.device,
            actions=acts.unsqueeze(0), skip_action_conditioning=False,
            num_inference_steps=self.steps, do_classifier_free_guidance=True,
            guidance_scale=CFG_GUIDANCE_SCALE, discretized_noise_level=self.noise_level,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return s

    def step(self, s, action):
        """一个动作 -> 一帧。返回 (png_bytes, 本帧耗时秒)。"""
        t0 = time.time()
        self.last_shapes = {"ctx": tuple(s["ctx"].shape),
                            "acts": tuple(s["acts"].shape)}
        a = torch.tensor([int(action)], dtype=torch.int64, device=self.device)
        s["acts"] = torch.cat([s["acts"][-self.buffer + 1:], a])
        tgt = next_latent(
            unet=self.unet, noise_scheduler=self.noise_scheduler,
            action_embedding=self.action_embedding,
            context_latents=s["ctx"].unsqueeze(0), device=self.device,
            actions=s["acts"].unsqueeze(0), skip_action_conditioning=False,
            num_inference_steps=self.steps, do_classifier_free_guidance=True,
            guidance_scale=CFG_GUIDANCE_SCALE,
            discretized_noise_level=self.noise_level,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_unet = time.time()
        s["ctx"] = torch.cat([s["ctx"][-self.buffer + 1:], tgt], dim=0)
        s["steps"] += 1
        png = to_png(self._decode(tgt))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.last_phases = {"unet_ms": round((t_unet - t0) * 1000, 1),
                            "decode_png_ms": round((time.time() - t_unet) * 1000, 1),
                            "tgt": tuple(tgt.shape)}
        return png, time.time() - t0


def to_png(img):
    """VAE 解出来的图 -> PNG bytes。

    形状不固定：diffusers 的 postprocess 在 output_type='pt' 下可能给 [B,C,H,W]
    也可能给 [C,H,W]，还可能是长度 1 的 list，所以这里一律归一化到 [C,H,W]。
    """
    if isinstance(img, (list, tuple)):
        img = img[0]
    t = img.detach().float()
    if t.dim() == 4:
        t = t[0]
    assert t.dim() == 3 and t.shape[0] in (1, 3), f"看不懂的图形状 {tuple(t.shape)}"
    a = t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    if a.shape[2] == 1:
        a = a.repeat(3, axis=2)
    buf = io.BytesIO()
    Image.fromarray((a * 255).round().astype(np.uint8)).save(buf, "PNG")
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/info"):
            eng = _CUR["engine"]
            if eng is None:
                return self._send({"loaded": None, "models": self.server.model_list,
                                   "actions": ACTIONS, "buffer": BUFFER_SIZE,
                                   "res": "320x256"})
            self._send({"actions": ACTIONS, "steps": eng.steps,
                        "device": str(eng.device),
                        "gpu": (torch.cuda.get_device_name(eng.device)
                                if eng.device.type == "cuda" else "cpu"),
                        "buffer": eng.buffer, "res": "320x256",
                        "bench": _CUR["bench"], "loaded": _CUR["id"],
                        "load_ms": _CUR["load_ms"],
                        "models": self.server.model_list})
        elif self.path.startswith("/status"):
            self._send(dict(_PROG, loaded=_CUR["id"]))
        elif self.path.startswith("/models"):
            self._send({"models": self.server.model_list, "loaded": _CUR["id"]})
        else:
            self._send({"error": "unknown"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        # 计算一律排到固定 GPU 线程，_send 在其外。曾经把发送放在 GPU 锁里，
        # 客户端中途断开时 wfile.write 阻塞，锁就再也放不出来——之后每个 /new 和
        # /step 永久挂住，而不碰锁的 /info 照样秒回，症状极具误导性。
        # 锁本身也带超时：宁可返回 503 让前端报错，也不要静默挂死。
        if self.path.startswith("/load"):
            mid = str(req.get("id", ""))
            if mid not in _MODELS:
                return self._send({"error": f"没有这个模型: {mid}",
                                   "known": list(_MODELS)}, 400)
            try:
                rb = bool(req.get("rebench"))
                cur = run_on_gpu(lambda: load_model_by_id(mid, self.server.args, rb),
                                 timeout=900)
            except Exception as e:
                return self._send({"error": f"{type(e).__name__}: {e}"}, 500)
            e2 = cur["engine"]
            return self._send({"loaded": cur["id"], "load_ms": cur["load_ms"],
                               "bench": cur["bench"], "steps": e2.steps, "buffer": e2.buffer,
                               "gpu": (torch.cuda.get_device_name(e2.device)
                                       if e2.device.type == "cuda" else "cpu")})

        eng = _CUR["engine"]
        if eng is None:
            return self._send({"error": "还没载入模型"}, 409)
        if self.path.startswith("/new"):
            try:
                s = run_on_gpu(lambda: eng.new_session(req.get("seed")))
            except TimeoutError:
                return self._send({"error": "busy"}, 503)
            sid = str(next(_IDS))                      # 单调计数，不能用 len(_STATE)
            while len(_STATE) >= 6:                    # 每个 session 一份显存
                _STATE.pop(next(iter(_STATE)))
            _STATE[sid] = s
            self._send({"id": sid, "frame": b64(s["png"]), "steps": 0})
        elif self.path.startswith("/step"):
            s = _STATE.get(str(req.get("id")))
            if s is None:
                return self._send({"error": "session expired"}, 410)
            try:
                png, dt = run_on_gpu(lambda: eng.step(s, req.get("action", 0)), timeout=90)
            except TimeoutError:
                return self._send({"error": "busy"}, 503)
            self._send({"frame": b64(png), "steps": s["steps"],
                        "ms": round(dt * 1000, 1),
                        "action": ACTIONS[int(req.get("action", 0))],
                        "phases": getattr(eng, "last_phases", None),
                        "shapes": getattr(eng, "last_shapes", None)})
        elif self.path.startswith("/selftest"):
            # 诊断用：在「请求线程」里跑同一段计算，和启动基准（主线程、serve_forever
            # 之前）对比。两者若差很多，问题就在请求上下文而不在模型或 GPU。
            eng = _CUR["engine"]
            import statistics
            def _run():
                s2 = eng.new_session(seed=0)
                for _ in range(2): eng.step(s2, 8)
                return [eng.step(s2, 8)[1] for _ in range(6)]
            ts = run_on_gpu(_run, timeout=300)
            self._send({"median_ms": round(statistics.median(ts) * 1000, 1),
                        "all_ms": [round(t * 1000, 1) for t in ts],
                        "startup_bench": _CUR["bench"]})
        else:
            self._send({"error": "unknown"}, 404)

    def log_message(self, *a):
        pass


def b64(png):
    return "data:image/png;base64," + base64.b64encode(png).decode()


WARMUP = 3          # 预热帧数：吃掉 cuDNN 自动调优 + GPU 升频的一次性开销


def warmup(eng):
    """只预热不计时。新形状的第一帧要 4.9 秒，不预热的话玩家按下第一个键会卡住。"""
    s = eng.new_session(seed=0)
    for i in range(WARMUP):
        eng.step(s, 8)
        progress("预热", i + 1, WARMUP)


def load_model_by_id(mid, args, rebench=False):
    """换模型：先把旧的从显存里彻底清掉，再载新的。

    不同时驻留多个：16GB 卡上每个约 4GB，留两个就没余量给推理时的中间张量了。
    换完之后所有旧 session 作废——它们的上下文是用旧 VAE 编出来的，混着用没意义。
    """
    m = _MODELS[mid]
    base = Path(args.base)
    t0 = time.time()
    progress("腾显存")
    if _CUR["engine"] is not None:
        _CUR["engine"] = None
        _STATE.clear()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    lat = next(iter(sorted((base / "latents").rglob("*.pt"))))
    progress("载入权重")
    eng = Engine(base / m["unet"], base / m["vae"], lat,
                 args.device, args.steps, args.noise_level)
    load_ms = round((time.time() - t0) * 1000)

    # 基准结果按 (模型, 显卡, 去噪步数, 噪声等级) 缓存到盘上。这些量不变，
    # 结果就不变，没道理每次换模型都重测一遍——实测那一步占换模型总耗时的一半以上。
    # 预热不能省（新形状的首帧要 4.9 秒），但预热本来就顺带完成了。
    key = f"{mid}|{eng.gpu_name}|{args.steps}|{args.noise_level}"
    cache = {}
    cpath = base / "bench_cache.json"
    if cpath.exists():
        try:
            cache = json.loads(cpath.read_text())
        except Exception:
            cache = {}
    if args.bench and key in cache and not rebench:
        progress("预热", 0, WARMUP)
        warmup(eng)
        bench = dict(cache[key], cached=True)
    elif args.bench:
        bench = benchmark(eng, args.bench)
        cache[key] = dict(bench, cached=False)
        cpath.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    else:
        progress("预热", 0, WARMUP)
        warmup(eng)
        bench = None
    _CUR.update(id=mid, engine=eng, load_ms=load_ms, bench=bench)
    progress("")
    return _CUR


def benchmark(eng, n=20):
    """预热 + 计时。第一帧含 CUDA 上下文和 kernel 自动调优，必须丢掉。"""
    progress("预热", 0, n + WARMUP)
    s = eng.new_session(seed=0)
    for i in range(WARMUP):
        eng.step(s, 8)
        progress("预热", i + 1, n + WARMUP)
    ts = []
    for i in range(n):
        ts.append(eng.step(s, 8)[1])
        progress("测速", i + 1 + WARMUP, n + WARMUP)
    ts = np.array(ts)
    r = {"n": n, "mean_ms": round(float(ts.mean() * 1000), 1),
         "p50_ms": round(float(np.median(ts) * 1000), 1),
         "min_ms": round(float(ts.min() * 1000), 1),
         "max_ms": round(float(ts.max() * 1000), 1),
         "fps": round(float(1 / ts.mean()), 2),
         "steps": eng.steps}
    print(f"[bench] {r}", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    base = HERE.parent / "outputs" / "gamengen"
    ap.add_argument("--base", default=str(base), help="权重根目录")
    ap.add_argument("--models", default=str(HERE / "models.json"))
    ap.add_argument("--model", default=None, help="启动时载哪个（默认清单第一个）")
    ap.add_argument("--latents", default=None,
                    help="种子 episode 的 .pt；默认取 <base>/latents 下第一个")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=4, help="每帧去噪步数（论文是 4）")
    ap.add_argument("--noise-level", type=int, default=9)
    ap.add_argument("--port", type=int, default=25677)
    # 默认只听回环。远端跑的时候只有 SSH 隧道进得来，不该对整个实验室网络敞开——
    # 这个接口没有任何认证，谁都能连上来占显存。本机跑时由 run.sh 显式传 0.0.0.0，
    # 因为浏览器是从别的机器访问这台机的 IP 的。
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--bench", type=int, default=20, help="启动时测多少帧，0=不测")
    a = ap.parse_args()

    spec = json.loads(Path(a.models).read_text())
    listed = [m for m in spec["models"]
              if (Path(a.base) / m["unet"] / "unet").exists()]
    for m in listed:
        _MODELS[m["id"]] = m
    assert listed, f"{a.base} 下一个模型都没有"

    threading.Thread(target=_worker, daemon=True).start()
    srv = ThreadingHTTPServer((a.bind, a.port), Handler)
    srv.args = a
    srv.model_list = [{k: m[k] for k in ("id", "name", "note", "repo", "github")} for m in listed]
    first = a.model or listed[0]["id"]
    print(f"[models] 可用 {[m['id'] for m in listed]}，先载 {first}", flush=True)
    run_on_gpu(lambda: load_model_by_id(first, a), timeout=900)
    print(f"[serve] http://{a.bind}:{a.port}  steps={a.steps}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
