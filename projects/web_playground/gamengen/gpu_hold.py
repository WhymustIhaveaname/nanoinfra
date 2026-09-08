#!/usr/bin/env python3
"""gpu_hold.py — 在 1202b 上占住一张卡的剩余显存，别让人往我们这张卡上挤。

1202b 是共享机器，八张 A6000。推理服务本身只吃 4–13 GB，剩下三十多 GB 是空的，
别人的任务看见空位就会挤上来，于是我们和他共享同一张卡的 SM，帧率被拖慢。
这个脚本把剩余显存吃掉，让别人的分配在这张卡上直接失败，自动去挑别的卡。

只占**我们已经在用的那张卡**。不要拿它去圈别人正在跑的卡。

    python gpu_hold.py --leave 20          # 留 20 GiB 给我们自己的服务，其余占住
    python gpu_hold.py --leave 20 --poll 30  # 每 30 秒补占一次新腾出来的显存

收到 SIGTERM / SIGINT 就释放退出。故意不占 SM：只 sleep，不做任何计算，
所以对别人的 nvidia-smi 利用率读数没有影响，也不浪费电。
"""

import argparse
import os
import signal
import sys
import time

GiB = 1024 ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leave", type=float, default=20.0,
                    help="留给自己人的显存 GiB（推理服务 + 换模型的峰值），默认 20")
    ap.add_argument("--poll", type=float, default=0,
                    help="每隔多少秒补占一次；0 = 只占一次就守着")
    ap.add_argument("--chunk", type=float, default=0.5, help="每次试探分配的块大小 GiB")
    a = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        sys.exit("没有可用的 CUDA 设备")

    # 用 CUDA_VISIBLE_DEVICES 选卡，和 doom_ngen_server.py 一致，
    # 这样「占位的卡」和「服务在跑的卡」不可能对不上。
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"[hold] {name} 共 {total / GiB:.1f} GiB "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')})",
          flush=True)

    blocks = []
    stop = False

    def bye(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)

    def grab():
        """试探着一块一块占，直到剩余量降到 --leave 以下。"""
        got = 0
        n = int(a.chunk * GiB) // 2      # float16，一个元素 2 字节
        while not stop:
            free, _ = torch.cuda.mem_get_info()
            if free <= a.leave * GiB:
                break
            try:
                blocks.append(torch.empty(n, dtype=torch.float16, device="cuda"))
                got += a.chunk
            except torch.cuda.OutOfMemoryError:
                # 别人抢在我们前面了，或者碎片化。停手，下一轮再试。
                break
        return got

    got = grab()
    free, _ = torch.cuda.mem_get_info()
    print(f"[hold] 已占 {got:.1f} GiB，这张卡还剩 {free / GiB:.1f} GiB 给我们自己用",
          flush=True)

    while not stop:
        time.sleep(a.poll if a.poll > 0 else 3600)
        if stop or a.poll <= 0:
            continue
        # 别人的任务结束会腾出显存，补占上去，否则空位又会被人挤进来
        more = grab()
        if more:
            free, _ = torch.cuda.mem_get_info()
            print(f"[hold] 补占 {more:.1f} GiB，剩 {free / GiB:.1f} GiB", flush=True)

    blocks.clear()
    torch.cuda.empty_cache()
    print("[hold] 已释放，退出", flush=True)


if __name__ == "__main__":
    main()
