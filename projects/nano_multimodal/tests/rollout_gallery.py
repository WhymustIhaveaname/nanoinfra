"""
rollout_gallery.py — what the video model GENERATES, for a human to look at.

decode_gallery.py renders the training DATA. Nothing in this project renders the
model's own output at length, so nobody has ever judged it: `val/video_ce 5.2476`
says the model beat `ln(64037) = 11.07`, and says nothing about whether the world
holds together when you walk around in it.

    python -m projects.nano_multimodal.tests.rollout_gallery
    python -m projects.nano_multimodal.tests.rollout_gallery --frames 24 --seeds 3

Writes outputs/gallery/rollout/ + a page that leads with what to look for.

WHAT THIS IS BUILT TO ANSWER, in the order the page asks it:

  1. DOES THE BUTTON DO ANYTHING? The decisive test is not "does turning look like
     turning" — that is a judgement call about a blurry 128px frame. It is a
     CONTROL: drive the SAME seed with two different action scripts under the same
     sampling seed, and compare. Identical codes mean the action tokens are being
     ignored and every "it looks controllable" impression was the prior. Different
     codes mean the signal is live, and only then is it worth asking whether the
     motion is the RIGHT motion.

  2. DOES IT SURVIVE A RE-ANCHOR? A row holds n_blocks generated frames; past that
     the session starts a new row carrying the tail's codes. Every clip here is
     driven well past that point, and the strip marks where it happened, because a
     world that dissolves exactly at the seam is a bug in the seam, not in the
     model.

  3. WHAT DECAYS, AND HOW FAST? Free-running autoregression compounds its own
     errors. The contact strip exists so this is read off one image instead of by
     replaying a GIF and trying to remember frame 4.

The seeds are real val clips, so frame 0 is ground truth and everything after it is
the model. The strip puts them side by side on purpose: the first tile is what the
world actually looked like.
"""

import argparse

import numpy as np

from projects.nano_multimodal import assembly, inference, spec
from projects.nano_multimodal.tests.decode_gallery import save_gif, save_strip

OUT = spec.GALLERY_DIR / "rollout"

# Action scripts, by id (spec.ACTION_NAMES). Each is (label, what to look for, ids).
# FWD and TL are the two whose effect on the frame is least ambiguous to the eye:
# forward changes scale, turning pans the whole scene sideways. NOOP is the control
# that says what the world does when nothing is pressed — if a still world drifts,
# the drift is the model's, not the player's.
SCRIPTS = [
    ("FWD",  "前进:整个场景应该向外放大、走廊尽头变近", 8),
    ("TL",   "左转:整幅画面应该横向平移", 0),
    ("NOOP", "什么都不按:世界应该基本静止 —— 它要是自己漂,那是模型在漂", 18),
]


def rollout(sess, action_id, n_frames, decode_px):
    """Drive one session n_frames steps on a fixed button; return (frames, codes,
    the frame indices where a re-anchor happened)."""
    from projects.nano_multimodal.decode.video import decode_pixels
    td = sess.shape["td"]
    frames, codes, seams = [], [], []
    before = sess.reanchors
    for i in range(n_frames):
        c = sess.step(action_id)
        if sess.reanchors > before:
            seams.append(i)
            before = sess.reanchors
        codes.append(c)
        if decode_px:
            flat = [x for f in sess.window for x in f]
            frames += list(decode_pixels(flat, device=sess.device)[-td:])
    return frames, codes, seams


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--frames", type=int, default=16,
                    help="latent frames per clip (each is td game frames)")
    ap.add_argument("--seeds", type=int, default=2, help="how many val clips to seed from")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    system, vocab = inference.load("video", args.ckpt, device=args.device)
    used = inference.checkpoints("video")[0] if args.ckpt is None else args.ckpt
    cfg = assembly.load_config("video")
    loader = assembly.build_loader("video", cfg, vocab, "val", args.device, batch_size=args.seeds)
    batch = next(iter(loader))
    shape = spec.video_shape()
    cpf, v_off = shape["codes_per_frame"], vocab.layout.offset(spec.VIDEO_TYPE_ID)

    def seed_codes(row):
        """The clip's FIRST latent frame — the given observation, ground truth."""
        ids = batch["idx"][row].cpu().numpy().tolist()
        first = next(i for i, t in enumerate(ids) if t >= v_off and t < v_off + spec.VIDEO_VOCAB)
        return [t - v_off for t in ids[first:first + cpf]]

    def new_session(codes, seed):
        return inference.Session(system, vocab, codes, seed=seed,
                                 temperature=args.temperature, top_k=args.top_k,
                                 device=args.device, sampler="static", carry=2)

    lines = [
        "# 视频世界模型:生成出来的是什么样",
        "",
        f"checkpoint `{used.name}`(run `{used.parent.name}`),val CE "
        f"**{inference._val_by_step(used.parent / 'metrics.jsonl').get(inference._step_of(used)):.4f}** "
        f"(起点 ln(64037) = 11.07)。温度 {args.temperature},top-k {args.top_k},"
        f"每段 {args.frames} 个 latent 帧 = {args.frames * shape['td']} 个游戏帧。",
        "",
        "每段的**第 0 帧是真实数据**(val 里一个片段的首帧),之后全部是模型自己写的。",
        "",
    ]

    # ---- 1. the control, and it comes first ---------------------------------
    # Two action scripts, ONE seed, ONE sampling seed. If the codes agree the action
    # band is decorative and every impression below is worthless, so this is decided
    # before anything is rendered.
    print("[1/2] control: same seed, two different buttons")
    base = seed_codes(0)
    _, codes_a, _ = rollout(new_session(base, 0), 8, args.frames, decode_px=False)   # FWD
    _, codes_b, _ = rollout(new_session(base, 0), 0, args.frames, decode_px=False)   # TL
    same = [int(np.array_equal(a, b)) for a, b in zip(codes_a, codes_b)]
    first_diff = next((i for i, s in enumerate(same) if not s), None)
    agree = sum(same)
    lines += [
        "## 1. 按键到底起不起作用",
        "",
        "同一个种子、同一个采样随机种子,只有按键不同(FWD 对 TL)。**如果两条轨迹的码完全一样,"
        "那动作 token 就是摆设**,下面所有\"看起来能操控\"的印象都不算数。",
        "",
        f"- {args.frames} 个 latent 帧里,**{agree} 帧的码完全相同**",
        f"- 第一处分歧在第 {first_diff} 帧" if first_diff is not None else
        "- **两条轨迹逐码相同 —— 动作没有起作用**",
        "",
        ("**判定:动作信号是活的。**" if agree < args.frames
         else "**判定:动作没有进入生成 —— 这是个 bug,先别看下面的图。**"),
        "",
    ]
    print(f"      {agree}/{args.frames} frames identical, first divergence at {first_diff}")

    # ---- 2. the clips -------------------------------------------------------
    # Per-clip pixel statistics, collected as the clips are rendered. Section 1 only
    # proves the button changes SOMETHING; this asks whether it changes the RIGHT
    # amount. It is model-free — mean |frame[t+1] - frame[t]| over the clip — so it
    # cannot inherit the model's own opinion of its output.
    motion = {}
    for si in range(args.seeds):
        base = seed_codes(si)
        lines += [f"## 2.{si + 1} 种子 {si}", ""]
        for label, look_for, aid in SCRIPTS:
            print(f"[2/2] seed {si} / {label}")
            sess = new_session(base, si)
            frames, _, seams = rollout(sess, aid, args.frames, decode_px=True)
            px = [(np.clip(f, 0, 1) * 255).astype(np.uint8) for f in frames]
            stem = f"s{si}_{label}"
            a = np.stack([f.astype(np.float32) for f in px])
            motion[stem] = (float(np.abs(a[1:] - a[:-1]).mean()),
                            float(np.abs(a[-1] - a[0]).mean()))
            save_gif(px, OUT / f"{stem}.gif", fps=10)
            picks = save_strip(px, OUT / f"{stem}.png", n=8)
            seam_note = (f"重锚发生在第 {', '.join(str(s) for s in seams)} 个 latent 帧"
                         if seams else "本段没有触发重锚")
            lines += [
                f"**{label}** — {look_for}",
                "",
                f"![{stem}]({stem}.png)",
                "",
                f"<sub>等距抽的 8 帧(游戏帧 {', '.join(str(p) for p in picks)});"
                f"{seam_note};动图:[{stem}.gif]({stem}.gif)</sub>",
                "",
            ]

    # The quantitative half of question 1, written after the clips exist.
    lines += ["## 3. 按键做的是不是【对的】那件事", "",
              "第 1 节只证明按键改变了【某些东西】。这一节问它改变的量对不对:"
              "不按键世界该基本不动,前进该慢慢改变,转身该把整幅画面横着推走 —— "
              "所以预期是 **NOOP < FWD < TL**。读数是逐帧像素差,不经过任何学出来的东西。",
              "", "| 片段 | 每帧平均变化 | 首尾差 |", "|---|---|---|"]
    for stem, (adj, ends) in motion.items():
        lines.append(f"| {stem} | {adj:.2f} | {ends:.1f} |")
    ok = all(motion.get(f"s{i}_NOOP", (9e9, 0))[0] < motion.get(f"s{i}_FWD", (0, 0))[0]
             < motion.get(f"s{i}_TL", (0, 0))[0] for i in range(args.seeds))
    lines += ["", ("**判定:三个按键的动量排序与预期一致(NOOP < FWD < TL),"
                   "每个种子上都成立。**" if ok else
                   "**判定:排序与预期不符 —— 按键有影响,但不一定是对的影响。**"), ""]

    lines += [
        "## 怎么读这一页",
        "",
        "第 1 节是**判据**,不是插图:它不问画面好不好看,只问按键有没有进到生成里去。"
        "只有它通过了,第 2 节里的每一段才值得用眼睛判断。",
        "",
        "第 2 节里看三件事:①按下的键和画面的变化对不对得上;"
        "②重锚那一帧有没有断裂 —— 世界要是正好在接缝处塌掉,那是接缝的 bug 不是模型的;"
        "③自由滚动几帧之后什么先烂掉。",
    ]
    (OUT / "README.md").write_text("\n".join(lines))
    print(f"\nWROTE {OUT / 'README.md'}")


if __name__ == "__main__":
    main()
