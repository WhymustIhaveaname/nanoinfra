"""
decode/text.py — ids -> string, with the seams left in.

The plain decode is one call. The teaching content is `units`: ONE ENTRY PER TOKEN,
carrying the exact substring that token contributed. Rendered with alternating
backgrounds, that shows a student what BPE actually did to their sentence — the most
valuable screen in the text line, and one that is invisible in any normal chat
interface.

BYTE RANGES, NOT INDEPENDENT FRAGMENTS. Byte-level BPE means a token's bytes need
not be a valid string on their own: a multi-byte character can straddle two tokens.
Decoding each token separately and concatenating therefore produces replacement
characters at exactly those boundaries — which is invisible on English and obvious
on CJK, i.e. the kind of bug a demo ships with. So the whole row is decoded once,
and each unit carries a [start, end) range into THAT string, computed by decoding
prefixes. A unit whose range is empty is a token that completed a character started
by its predecessor; the browser draws it as a zero-width tick, which is the honest
picture.
"""

from projects.nano_multimodal.decode import Rendered, band_of, spans_of


def render(row, vocab, **_):
    tok = vocab.tokenizers["text"]
    ids = [int(x) for x in row]

    text_ids, units = [], []
    prefix_len = 0
    for pos, tid in enumerate(ids):
        band, type_id, local = band_of(vocab, tid)
        if band != "text":
            # A control token has no text; it still gets a unit so the browser can
            # line units up with positions one-to-one.
            units.append({"tokens": [pos, pos + 1], "range": [prefix_len, prefix_len],
                          "band": band, "preview": vocab.resolver.name_of(tid) or ""})
            continue
        text_ids.append(tid)
        grown = len(tok.decode(text_ids))
        units.append({"tokens": [pos, pos + 1], "range": [prefix_len, grown],
                      "band": "text", "preview": tok.decode(text_ids)[prefix_len:grown]})
        prefix_len = grown

    text = tok.decode(text_ids) if text_ids else ""
    return Rendered(
        kind="text",
        spans=spans_of(row, vocab),
        units=units,
        media=text,
        notes=[f"{len(text_ids)} text tokens -> {len(text)} characters "
               f"({len(text)/max(len(text_ids),1):.1f} chars per token)"],
    )
