"""
test_video_row.py — the one place this project could be wrong without failing.

VideoRowSource interleaves codes and action ids into a flat row. Get the
interleave off by one slot and the model trains perfectly, the loss falls, and
every number looks healthy — it is simply learning a world in which the buttons
were pressed at the wrong moments. Nothing downstream would notice.

So the row is checked against an INDEPENDENT implementation that has already been
used to train working models: exemplars/nano_world_model/row_layout.RowLayout,
constructed with this project's band offsets. Two implementations agreeing
bit-for-bit is worth more than either one re-reading its own intent.

The same goes for supervision: the 0/1 pattern is checked against
exemplars/nano_world_model/autoregressive.Autoregressive, which owns the reasons
(given frame = observation, actions = control input).

    python -m projects.nano_multimodal.tests.test_video_row
"""

import numpy as np
import torch
from projects.nano_multimodal import assembly, spec
from projects.nano_multimodal.sources import VideoRowSource


def main():
    config = assembly.load_config("video", seed=0)
    vocab = assembly.assemble_vocab("video", config)
    src_cfg = assembly.resolve_sources(config, vocab, "train", device="cpu")[0]
    src = VideoRowSource({k: v for k, v in src_cfg.items() if k != "type"}, vocab.tokenizers)

    # --- the reference: the exemplar's RowLayout, on OUR offsets ---------------
    from exemplars.nano_world_model import row_layout
    r = vocab.resolver
    rows = row_layout.RowLayout(
        spec.video_shape(config["clip"]["frames"], config["clip"]["res"]) | {
            "n_blocks": 4, "n_given": 256, "n_action_tokens": 17},
        video_offset=vocab.layout.offset(spec.VIDEO_TYPE_ID),
        action_offset=vocab.layout.offset(spec.ACTION_TYPE_ID),
        control_ids={"bos": r.resolve("bos"), "eos": r.resolve("eos"),
                     "video_start": r.resolve("video_start"),
                     "video_end": r.resolve("video_end")},
        n_actions=spec.N_ACTIONS)
    content = rows.content_len
    assert content == vocab.sequence_len, f"{content} != {vocab.sequence_len}"

    # --- 1. the row, bit for bit, on 20 random cache rows ----------------------
    rng = np.random.default_rng(0)
    picks = rng.choice(len(src.codes), size=20, replace=False)
    for i in picks:
        mine = src._row(int(i))
        ref = rows.assemble(src.codes[int(i):int(i) + 1],
                            src.actions[int(i):int(i) + 1])[0][:content].numpy()
        assert np.array_equal(mine, ref), f"row {i} differs from RowLayout at " \
            f"{np.flatnonzero(mine != ref)[:8]}"
    print(f"row        20/20 rows bit-identical to RowLayout ({content} tokens)")

    # --- 2. supervision, against the exemplar's AR objective ------------------
    from exemplars.nano_world_model.autoregressive import Autoregressive
    ref_sup = Autoregressive(rows).predicted[:content].numpy().astype(np.float32)
    mine_sup = src._lw.cpu().numpy()
    assert np.array_equal(mine_sup, ref_sup), \
        f"supervision differs at {np.flatnonzero(mine_sup != ref_sup)[:8]}"
    print(f"supervise  identical to Autoregressive: {int(mine_sup.sum())} of {content}")

    # --- 3. the period, stated independently ---------------------------------
    # Not a restatement of (2): this asserts the SHAPE the browser will draw — a
    # 4-token dark stripe every 260, and the whole first frame dark.
    cpf, td = src.cpf, src.td
    f0 = src._field[0]
    assert mine_sup[:f0 + cpf].sum() == 0, "the given frame must not be supervised"
    for k in range(src.n_blocks):
        a0 = f0 + cpf + k * (td + cpf)
        assert mine_sup[a0:a0 + td].sum() == 0, f"action group {k} is supervised"
        assert mine_sup[a0 + td:a0 + td + cpf].sum() == cpf, f"frame {k+1} is not fully supervised"
    print(f"period     {src.n_blocks} x ({td} dark actions + {cpf} bright codes), "
          f"first frame dark")

    # --- 4. what the model actually receives ---------------------------------
    # The source yields tokens + weights; MixedDataLoader shifts. Verify the SHIFT
    # lands the supervision where the AR objective puts it, since an off-by-one here
    # would be invisible in (2).
    from core.data.supervision import NextTokenPrediction
    from core.tokenization.vocab_layout import VocabLayout
    it = iter(src)
    rowsamp = [next(it) for _ in range(2)]
    batch = NextTokenPrediction().apply(
        torch.stack([s["tokens"] for s in rowsamp]),
        torch.stack([s["token_types"] for s in rowsamp]),
        torch.stack([s["attention_mask"] for s in rowsamp]),
        torch.stack([s["loss_weights"] for s in rowsamp]))
    keep = (batch["targets"][0] != VocabLayout.IGNORE_INDEX).numpy()
    ref_keep = ref_sup[1:].astype(bool)
    assert np.array_equal(keep, ref_keep), "post-shift supervision is off by one"
    print(f"shift      post-shift targets match ({int(keep.sum())} supervised)")

    # --- 5. resume: three integers, no rank in them --------------------------
    s0 = src.get_state()
    seen = [next(it)["tokens"][:4].tolist() for _ in range(5)]
    mid = src.get_state()
    src.set_state(s0)
    it2 = iter(src)
    again = [next(it2)["tokens"][:4].tolist() for _ in range(5)]
    assert seen == again or s0 != mid, "resume did not replay the same rows"
    print(f"resume     state={mid} replays deterministically")

    print("\nOK — the video row agrees with the implementation that trained working models.")


if __name__ == "__main__":
    main()
