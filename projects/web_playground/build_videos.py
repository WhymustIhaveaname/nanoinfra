"""build_videos.py — 从每段录像里截 1 分钟转成 MP4，供预览页播放。

和 build_preview.py 分工不同：那边是 17 帧训练切片的「原始 vs 重建」逐帧对照，
必须无损（PNG 精灵图），因为要比的就是编解码损失。这边只是「这盘游戏长什么样」，
走视频容器，H.264 的量化在这个用途下无所谓。

只截 SECONDS 秒，不放整集：整集 4000–21000 帧，传输量吃不消。
从第 0 帧开始，所以开头那两秒的传送雾（9 个角色挤在同一出生点）也在里面。

输出两路：
  raw    240x320 原生 —— 录制器真正存下来的
  model  128x128     —— 同一段被 resize 成模型实际看到的样子
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exemplars.nano_world_model import spec              # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "media"
FPS = 35                 # Doom 的 tic 率
SECONDS = 60
NFRAMES = FPS * SECONDS  # 2100


def encode(frames, w, h, dst):
    """把 uint8 帧流喂给 ffmpeg。rawvideo 管道，不落中间文件。"""
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    for f in frames:
        p.stdin.write(f.tobytes())
        n += 1
    p.stdin.close()
    assert p.wait() == 0, f"ffmpeg failed on {dst}"
    return n


def main():
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for tag in ("bots", "bots_long", "pans"):
        shard = spec.PIXEL_SHARD_DIR / f"pv_{tag}_0000.bin"
        sc_path = spec.SIDECAR_DIR / f"pv_{tag}_0000_sc.npz"
        if not shard.exists() or not sc_path.exists():
            print(f"  {tag}: 缺像素分片或 sidecar，跳过")
            continue
        sc = np.load(spec.SIDECAR_DIR / f"pv_{tag}_0000_sc.npz", allow_pickle=True)
        h, w = int(sc["h"]), int(sc["w"])
        px = np.memmap(shard, dtype=np.uint8, mode="r").reshape(-1, h, w, 3)
        total = len(px)
        take = min(NFRAMES, total)

        raw = f"{tag}_raw.mp4"
        n = encode((np.asarray(px[i]) for i in range(take)), w, h, OUT / raw)

        model, R = f"{tag}_model.mp4", spec.RES
        encode((np.asarray(Image.fromarray(np.asarray(px[i])).resize((R, R),
                                                                    Image.BILINEAR))
                for i in range(take)), R, R, OUT / model)

        mb = ((OUT / raw).stat().st_size + (OUT / model).stat().st_size) / 1e6
        out[tag] = {"raw": f"outputs/media/{raw}", "model": f"outputs/media/{model}",
                    "frames": int(n), "seconds": round(n / FPS, 1),
                    "episode_frames": int(total),
                    "episode_seconds": round(total / FPS, 1),
                    "raw_res": f"{w}x{h}", "model_res": f"{R}x{R}",
                    "mb": round(mb, 1)}
        print(f"  {tag}: 截 {n} 帧 / 整集 {total} 帧, {mb:.1f} MB")

    (OUT.parent / "videos.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> videos.json ({len(out)} 段)")


if __name__ == "__main__":
    main()
