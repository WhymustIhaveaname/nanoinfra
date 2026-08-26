"""
decode/ — token -> 世界. The other half of the tokenizer we were handed.

WHY THIS IS A TOP-LEVEL PACKAGE AND NOT `serve/decoders.py`
-----------------------------------------------------------
Because the dependency arrow only runs one way. `serve/` needs decode; decode
needs nothing from `serve/`. A module that a CLI must import in order to write a
GIF has no business living inside a web server's package — and there are four
callers, only two of which are web:

    serve/browse.py       what the training data looks like
    serve/play.py         what the model produced
    scripts / notebooks   dump a gallery of GIFs, no HTTP involved
    tests                 round-trip: encode a known clip, decode it, compare

Placed under serve/, the third and fourth callers would have to import a web
server to render a picture, and the first person who wants a decoder without a
browser would copy it instead. Placed here, it is what it actually is: the inverse
of sources.py. sources.py is 世界 -> token; this is token -> 世界; the model in
between never sees anything but integers. The directory listing says so.

THE CONTRACT
------------
One question, asked the same way for every line:

    render(row, assembled) -> Rendered

`row` is a row of GLOBAL ids as the loader produced it — or as the model just
generated, which is the point: the browser and the inference web call the SAME
function, so any difference a student sees between training data and model output
comes from the model, never from the renderer.

`assembled` carries the layout, so this package never hardcodes a band offset: it
asks the layout which band an id is in, exactly as the model's token_types were
derived.

Rendered is JSON-safe except for `media`:

    kind      "text" | "motion" | "video"
    spans     [{start, end, band, type_id, label}]   segment map for the row anatomy
    units     [{tokens: [i, j), preview}]            the atoms a human perceives:
                                                     one word-piece / one 4-frame
                                                     motion step / one 8x8 pixel block
    media     PNG/WebP bytes, or a string for text
    notes     one-line human explanations, shown verbatim

`units` is the field that earns this package its existence. It is what lets the
browser answer "what does THIS integer mean" by highlighting a word-piece, a
quarter-second of movement, or an 8x8 patch of one frame — same question, same
interaction, three modalities.

A note on the word "decode": for motion this chains codec.decode -> forward
kinematics -> rasterisation, which is more than a tokenizer inverse. It is still
decode from the CALLER's point of view — the caller asks what a row MEANS, not for
FK to be run. The chaining is this package's business.

Imports are lazy, per line: the text line must not need the Cosmos decoder (~86MB
TorchScript), and the video line must not need the SMPL body model.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Rendered:
    kind: str
    spans: List[Dict[str, Any]] = field(default_factory=list)
    units: List[Dict[str, Any]] = field(default_factory=list)
    media: Optional[Any] = None
    notes: List[str] = field(default_factory=list)


def get(line: str):
    """line name -> that line's render(row, vocab). Lazy import, so a caller only
    pays for the modality it asks about (the Cosmos decoder is ~86MB of
    TorchScript; the motion path needs the SMPL body model)."""
    import importlib
    if line not in ("text", "motion", "video"):
        raise KeyError(f"unknown line {line!r}")
    return importlib.import_module(f"projects.nano_multimodal.decode.{line}").render


def render(line, row, vocab, **kw):
    """The one call. `row` is a list/array of GLOBAL ids — from the dataset or from
    the model, and deliberately indistinguishable here."""
    return get(line)(row, vocab, **kw)


def band_of(vocab, token_id: int):
    """global id -> (band name, type_id, local id). Asks the LAYOUT, so this module
    never hardcodes an offset."""
    from projects.nano_multimodal.assembly import TYPE_IDS
    tid = int(vocab.layout.classify_token_types(_t(token_id)).item())
    for name, t in TYPE_IDS.items():
        if t == tid and t in vocab.layout.ranges:
            return name, tid, int(token_id) - vocab.layout.offset(t)
    return None, tid, int(token_id)


def _t(x):
    import torch
    return torch.as_tensor([int(x)])


def spans_of(row, vocab) -> List[Dict[str, Any]]:
    """Segment map for ANY row: consecutive runs of the same band, labelled by band
    name — and, for a run of length 1 in the control band, by the control token's
    readable name (`video_start`, not `ctrl0`).

    Computed from the layout alone. That is what makes ONE browser component
    correct for all three lines: the band structure of a row is a fact of the
    vocabulary, not of the modality.
    """
    import numpy as np
    import torch

    from projects.nano_multimodal.assembly import TYPE_IDS

    ids = np.asarray(row, dtype=np.int64)
    types = vocab.layout.classify_token_types(torch.from_numpy(ids)).numpy()
    by_type = {v: k for k, v in TYPE_IDS.items()}

    # In a CONTENT band a run is a run of the band; in the CONTROL band it is a run
    # of the same token. bos and text_start sit next to each other and mean
    # different things, and a packed text stream carries a bos at every DOCUMENT
    # boundary — each deserves its own marker. But a padding tail of two hundred
    # identical eos is one fact, not two hundred, and splitting it would bury the
    # row's real structure in the browser.
    ctrl = TYPE_IDS["control"]
    spans, start = [], 0
    for i in range(1, len(types) + 1):
        if i == len(types):
            cut = True
        elif types[i] != types[start]:
            cut = True
        elif types[start] == ctrl:
            cut = ids[i] != ids[start]
        else:
            cut = False
        if not cut:
            continue
        tid = int(types[start])
        band = by_type.get(tid, f"type{tid}")
        label = band
        if tid == ctrl:
            label = vocab.resolver.name_of(int(ids[start])) or band
        spans.append({"start": start, "end": i, "band": band,
                      "type_id": tid, "label": label})
        start = i
    return spans
