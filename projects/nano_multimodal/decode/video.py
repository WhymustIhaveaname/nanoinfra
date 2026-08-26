"""
decode/video.py — codes -> frames, and one integer -> one 8x8 patch.

    global ids -> local codes -> [n_latent, 16, 16] -> Cosmos DV4x8x8 decoder
                -> pixels [frames, 128, 128, 3] -> WebP

The codec is spatial /8 and temporal /4, so a row's 1280 codes are five 16x16 grids
and each code owns an 8x8 patch of one latent frame. `units` carries that mapping:
token index -> (latent frame, grid row, grid col, pixel box). It is what lets the
browser highlight the exact patch a clicked integer is responsible for — the hardest
thing about a video tokenizer to explain in words and the easiest to show.

The action ids interleaved between frames decode to NAMES, not pixels. They appear
in `spans` as their own band and in `notes` as the button sequence that drove the
clip.

Loads the ~86MB decoder TorchScript on first use and keeps it; the encoder is never
needed here (the data was encoded offline), which halves the residency. Lazy import
throughout: the text and motion lines must not pay for any of this.

⚠ THE DECODER IS NOT DETERMINISTIC. The same 256 codes decoded twice differ by up to
1.4e-2 (mean 6e-4) on this box — bigger than the difference between decoding a frame
alone and decoding it inside its clip. Nothing visible follows from that, but two
things do: no pixel-exact test of a rendered frame is possible, and every comparison
of two rendered frames sits on top of that noise floor. tests/test_inference.py
measures and prints it rather than assuming a number.
"""

import numpy as np

from projects.nano_multimodal import spec
from projects.nano_multimodal.decode import Rendered, band_of, spans_of

_DEC = {}


def _decoder(device="cuda"):
    """The frozen Cosmos decoder, one per device, loaded once."""
    import torch
    if device not in _DEC:
        path = spec.VIDEO_CODEC_DIR / "decoder.jit"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — the video decoder is what turns codes back into "
                f"pixels. Fetch it with exemplars/nano_world_model/data/download.py.")
        _DEC[device] = torch.jit.load(str(path)).to(device).eval()
    return _DEC[device]


def decode_pixels(local_codes, res=spec.RES, device="cuda"):
    """local codes -> [T, H, W, 3] float in [0,1]. Ported from the research line's
    codec wrapper: reshape to [B, n_latent, s, s], decode, clamp to [0,1]."""
    import torch
    s = res // spec.CODEC_SPATIAL_DS
    c = np.asarray(local_codes, dtype=np.int64)
    t = len(c) // (s * s)
    idx = torch.from_numpy(c[:t * s * s]).reshape(1, t, s, s).int().to(device)
    with torch.no_grad():
        rec = _decoder(device)(idx)[0].float()          # [3, T', H, W]
    rec = rec.permute(1, 2, 3, 0)                        # [T', H, W, 3]
    return (rec.clamp(-1, 1) * 0.5 + 0.5).cpu().numpy()


def render(row, vocab, device="cuda", res=spec.RES, **_):
    ids = np.asarray(row, dtype=np.int64)
    shape = spec.video_shape(res=res)
    cpf, s = shape["codes_per_frame"], res // spec.CODEC_SPATIAL_DS

    codes, actions, units = [], [], []
    for pos, tid in enumerate(ids):
        band, _type_id, local = band_of(vocab, int(tid))
        if band == "video":
            k = len(codes)                               # index within the code stream
            frame, cell = divmod(k, cpf)
            gy, gx = divmod(cell, s)
            units.append({
                "tokens": [pos, pos + 1], "band": "video", "code": local,
                "frame": frame, "grid": [gy, gx],
                # The pixel box this ONE integer is responsible for.
                "box": [gx * spec.CODEC_SPATIAL_DS, gy * spec.CODEC_SPATIAL_DS,
                        spec.CODEC_SPATIAL_DS, spec.CODEC_SPATIAL_DS],
                "preview": f"frame {frame}, patch ({gy},{gx})",
            })
            codes.append(local)
        elif band == "action":
            name = spec.ACTION_NAMES[local] if local < len(spec.ACTION_NAMES) else str(local)
            units.append({"tokens": [pos, pos + 1], "band": "action",
                          "action": local, "preview": name})
            actions.append(name)
        else:
            units.append({"tokens": [pos, pos + 1], "band": band,
                          "preview": vocab.resolver.name_of(int(tid)) or ""})

    frames = decode_pixels(codes, res=res, device=device) if codes else np.zeros((0, res, res, 3))
    return Rendered(
        kind="video",
        spans=spans_of(row, vocab),
        units=units,
        media=(frames * 255).astype(np.uint8),
        notes=[f"{len(codes)} codes = {len(codes)//cpf} latent frames of {s}x{s}; "
               f"one code = one {spec.CODEC_SPATIAL_DS}x{spec.CODEC_SPATIAL_DS} patch",
               "buttons: " + " ".join(actions) if actions else "no action tokens"],
    )
