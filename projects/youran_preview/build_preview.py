"""build_preview.py — 生成数据预览页：每段 clip 并排显示 原始 vs encode->decode 重建。

为什么是 PNG 精灵图而不是 GIF/MP4：这个页面的唯一用途是肉眼比较编解码损失，
任何有损容器（GIF 的 256 色调色板、H.264 的量化）都会自己引入失真，把要看的东西盖住。
一段 clip = 一张横向拼接的无损 PNG（17 帧 × 128px），浏览器用 CSS steps() 逐帧播放。

数据来源分两类，各自走仓库里真实的那条读取路径：
  * 下载的 PPO 语料 -> a16z parquet（10 帧滑窗，展开成连续 run 后切 clip）
  * 本地录制       -> data/record 写的像素 shard + sidecar（encode.recorded_clips）
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exemplars.nano_world_model import spec                      # noqa: E402
from exemplars.nano_world_model.data import encode as enc        # noqa: E402
from exemplars.nano_world_model.data.codec import CosmosDV       # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
MEDIA = OUT / "media"
FRAMES, RES = spec.FRAMES, spec.RES
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 录制分层：tag -> (中文名, 一句话说明)。这四个（含下载的 PPO）才是对等的对象。
RECORDED = [
    ("bots",      "bots",
     "8 个电脑玩家打死斗，每集 4000 帧（约 114 秒）。世界自己会动，提供 dynamics。"),
    ("bots_long", "bots_long",
     "和 bots 同一个世界、同一个策略，唯一区别是每集 21000 帧（约 10 分钟）——"
     "为切 129 帧长窗口留余地。"),
    ("pans",      "pans",
     "原意是「摇镜头」：该在静态世界里把「按键→画面」教干净。但配方写死 "
     "world_bots_frac: 1.0，静态世界那条路在 run_episode 里会抛异常，"
     "所以这一层实际退化成了和 bots 完全一样。"),
]


# --- 下载语料：只读前若干个 batch，不把整个 1GB shard 读进内存 -------------------
def parquet_clips_head(path, n_clips, max_rows=400):
    """从 a16z parquet 的前 max_rows 个滑窗里切出 n_clips 段。

    encode.parquet_clips 会先把整个 shard 读进一个 dict 再 yield，1GB 文件下
    内存吃不消；预览只要开头几段，所以这里复刻它的索引逻辑但提前收手。
    """
    import pyarrow.parquet as pq

    by_step, ep_seen = {}, None
    for batch in pq.ParquetFile(path).iter_batches(
            batch_size=64, columns=["images", "actions", "episode_id", "step_id"]):
        d = batch.to_pydict()
        for im, ac, ep, st in zip(d["images"], d["actions"], d["episode_id"], d["step_id"]):
            if ep_seen is None:
                ep_seen = ep
            if ep != ep_seen:                     # 只取第一个 episode，避免跨集
                continue
            by_step[st] = (im, ac)
        if len(by_step) >= max_rows:
            break

    # 全局帧号 -> (滑窗起点, 窗内偏移)。窗与窗在重叠处逐像素一致，取谁都一样。
    src = {}
    for st in sorted(by_step):
        for k in range(enc.WINDOW):
            src[st + k] = (st, k)

    present = sorted(src)
    runs, start = [], 0
    for i in range(1, len(present) + 1):
        if i == len(present) or present[i] != present[i - 1] + 1:
            runs.append(present[start:i])
            start = i
    runs = [r for r in runs if len(r) >= FRAMES]
    runs.sort(key=len, reverse=True)

    out = []
    for run in runs:
        for g0 in range(0, len(run) - FRAMES + 1, FRAMES):
            clip, acts = [], []
            for g in run[g0:g0 + FRAMES]:
                st, k = src[g]
                im, ac = by_step[st]
                clip.append(enc._decode_png(im[k], RES))
                acts.append(ac[k])
            out.append((np.stack(clip), np.asarray(acts, np.uint8)))
            if len(out) >= n_clips * 6:       # 多取一些，再均匀挑
                return spread(out, n_clips)
    return spread(out, n_clips)


def spread(items, n):
    """从序列里均匀取 n 个。取开头 n 个是错的：一集 400 帧里前 34 帧往往还在
    同一个动作段内，不同配方的录像在那一段可能完全重合（mixed 和 atk 第一版
    PSNR 一模一样就是这么来的）。"""
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[min(len(items) - 1, int(i * step + step / 2))] for i in range(n)]


def recorded_clips_head(tag, n_clips):
    """从某个 tag 的像素 shard 里均匀取 n_clips 段，走 encode.recorded_clips 本人。"""
    # 分片号是纯数字，必须这么写死：`pv_bots_*` 会把 pv_bots_long_0000.bin 也吞进来。
    shards = sorted(spec.PIXEL_SHARD_DIR.glob(f"pv_{tag}_[0-9][0-9][0-9][0-9].bin"))
    if not shards:
        return []
    out = []
    for sh in shards:
        for clip, acts, _split in enc.recorded_clips(str(sh), FRAMES, RES, stride=FRAMES):
            out.append((clip, acts))
    return spread(out, n_clips)


# --- 编解码 -------------------------------------------------------------------
_DEC = None


def decoder():
    global _DEC
    if _DEC is None:
        _DEC = torch.jit.load(str(spec.CODEC_DIR / "decoder.jit")).to(DEVICE).eval()
    return _DEC


def roundtrip(clips, codec):
    """[(clip,acts)] -> (codes[B,1280], recon[B,17,128,128,3] uint8)"""
    codes = codec.encode([c for c, _ in clips])                  # [B, t*h*w] int32
    s = RES // spec.CODEC_SPATIAL_DS
    t = codes.shape[1] // (s * s)
    idx = torch.from_numpy(codes.astype(np.int64)).reshape(-1, t, s, s).int().to(DEVICE)
    with torch.no_grad():
        out = decoder()(idx)
    # decoder.jit 直接返回张量 [B,3,T,H,W]；个别版本包一层 tuple，两种都接住。
    rec = (out if torch.is_tensor(out) else out[0]).float()
    rec = rec.permute(0, 2, 3, 4, 1).clamp(-1, 1) * 0.5 + 0.5
    return codes, (rec.cpu().numpy() * 255).round().astype(np.uint8)


def sprite(frames, path):
    """把 [T,H,W,3] 横向拼成一张无损 PNG。"""
    T, H, W, _ = frames.shape
    sheet = Image.new("RGB", (W * T, H))
    for i in range(T):
        sheet.paste(Image.fromarray(frames[i]), (i * W, 0))
    sheet.save(path, "PNG", optimize=True)


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    codec = CosmosDV(str(spec.CODEC_DIR), device=DEVICE)
    groups, n_clips = [], 4

    # 1) 下载的 PPO 语料
    pq_files = sorted(spec.PIXEL_PARQUET_DIR.glob("*.parquet"))
    if pq_files:
        clips = parquet_clips_head(pq_files[0], n_clips)
        if clips:
            groups.append(("downloaded", "PPO（下载）",
                           f"a16z 公开语料 {pq_files[0].name}，别人训好的 PPO 智能体录的；"
                           "10 帧滑窗展开成连续 run 后切 17 帧", clips))

    # 2) 本地录制的各变体
    for tag, title, note in RECORDED:
        clips = recorded_clips_head(tag, n_clips)
        if clips:
            groups.append((f"rec_{tag}", title, note, clips))

    # 3) 编解码 + 落盘
    manifest = []
    for gid, title, note, clips in groups:
        codes, recon = roundtrip(clips, codec)
        items = []
        for i, (clip, acts) in enumerate(clips):
            a = f"{gid}_{i}_orig.png"
            b = f"{gid}_{i}_recon.png"
            sprite(clip, MEDIA / a)
            sprite(recon[i], MEDIA / b)
            diff = np.abs(clip.astype(np.int16) - recon[i].astype(np.int16))
            mse = float((diff.astype(np.float64) ** 2).mean())
            psnr = float(10 * np.log10(255.0 ** 2 / mse)) if mse > 0 else float("inf")
            items.append({
                "orig": f"outputs/media/{a}", "recon": f"outputs/media/{b}",
                "mae": round(float(diff.mean()), 2),
                "frames": int(clip.shape[0]), "res": int(clip.shape[1]),
                "actions": [spec.ACTION_NAMES[int(x)] for x in acts],
                "n_codes": int(codes.shape[1]),
                "uniq_codes": int(len(np.unique(codes[i]))),
                "psnr": round(psnr, 2),
            })
        manifest.append({"id": gid, "title": title, "note": note, "clips": items})
        print(f"  {gid}: {len(items)} clips, PSNR "
              f"{[c['psnr'] for c in items]}", flush=True)

    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(manifest)} groups -> {OUT/'manifest.json'}")


if __name__ == "__main__":
    main()
