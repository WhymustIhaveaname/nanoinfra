"""Fetch FineWeb sample-10BT parquet shards into <base>/base_data/.

Two uses, and the default serves the first:

  from scratch   no arguments -> the six shards this exemplar is documented
                 against (~13 GB on disk). Shard COUNT is not a detail: the val
                 split is DECLARED (train_text.yaml pins one file by name) and
                 train is `rest: true`, so the training corpus is whatever else
                 you happened to fetch. Fetch fewer and the run silently repeats
                 data instead of erroring — the orchestrator now prints the
                 resulting epoch count at startup. See README "Data".
  incrementally  pass shard ids to add more; they join TRAIN. The
                 `shard_NNN_00000.parquet` prefix keeps the listing sorted and
                 stable. Moving the val ruler is a config edit, never a
                 side effect of downloading (modalities/text/datasets.py: splits
                 are declared, never inferred).

Downloads go through huggingface_hub.hf_hub_download into a _hf/ cache beside
the shards, are verified readable, then moved into place.

Run: .venv/bin/python exemplars/text_pretrain/data/download_shards.py
     .venv/bin/python exemplars/text_pretrain/data/download_shards.py 006 007
"""
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from core.utils import get_base_dir

REPO = "HuggingFaceFW/fineweb"
# Resolved the same way every reader resolves it (core.utils.get_base_dir, honouring
# NANOINFRA_BASE_DIR) rather than hardcoded: this script WRITES the directory that
# fineweb.list_parquet_files READS, and a writer that computes the path differently
# from its reader is a bug that only shows up for whoever relocates outputs/.
BASE = Path(get_base_dir()) / "base_data"
CACHE = BASE / "_hf"


def main(idxs):
    for idx in idxs:
        hf_name = f"{idx}_00000.parquet"
        dst = BASE / f"shard_{idx}_00000.parquet"
        if dst.exists():
            print(f"[skip] {dst} already exists")
            continue
        print(f"[download] sample/10BT/{hf_name} ...", flush=True)
        local = hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=f"sample/10BT/{hf_name}",
            local_dir=str(CACHE),
        )
        # verify integrity before publishing to base_data
        f = pq.ParquetFile(local)
        nrows = f.metadata.num_rows
        assert nrows > 0, f"empty parquet {local}"
        print(f"[verify] {hf_name}: rows={nrows:,} row_groups={f.num_row_groups} OK", flush=True)
        shutil.move(local, dst)
        print(f"[done] -> {dst}", flush=True)
    print("ALL DOWNLOADS COMPLETE", flush=True)


if __name__ == "__main__":
    # The documented set, not a subset of it: README and RESULTS both describe a
    # six-shard corpus (5 train + 1 val = single-epoch at the Chinchilla budget).
    # The old default of 003-005 gave a fresh clone two train shards, ~40% of what
    # the docs claim, and turned the champion run into ~1.87 epochs without a word
    # in the log.
    main(sys.argv[1:] or ["000", "001", "002", "003", "004", "005"])
