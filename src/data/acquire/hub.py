"""Live Hugging Face Hub downloader for real acquisition runs.

This module exists for later real use; **no test in this repository imports
or calls it** — the test suite stubs the :class:`~src.data.acquire.acquire.Downloader`
protocol instead. Network access happens only through :func:`hub_downloader`
and :func:`main`.

Typical use (not run in CI)::

    .venv/bin/python -m src.data.acquire.hub --source cais/mmlu \
        --output-root data/raw

The Hub metadata object (``datasets.get_dataset_infos`` / ``HfApi.dataset_info``)
is passed straight into :func:`~src.data.acquire.acquire.acquire_source`, which
reads the license from it; nothing here decides admission.

Source registry: the admitted sources from ``docs/dataset-strategy.md`` §2
with their declared row counts. Licenses are *not* hard-coded — they are
resolved from the Hub card at acquisition time, and anything missing is
quarantined by the manifest logic. Sources whose cards declare no license
today (HellaSwag, SHP) quarantine automatically. UltraFeedback is the
designated OOD set: only its binarized subset may ever be fetched, and it
yields zero training rows; that policy lives in the splits/mixture layers,
this module only streams.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from .acquire import MAX_CORPUS_BYTES, Downloader, acquire_source
from .manifest import STATUS_QUARANTINE, SourceManifest

#: Admitted sources from the dataset strategy, with declared Hub row counts
#: (2026-11-19). Acquisition respects each declared cap via ``max_rows``.
_SOURCE_REGISTRY: dict[str, dict[str, Any]] = {
    "cais/mmlu": {
        "declared_rows": 231_400,
        "split": "auxiliary_train",
        "max_rows": 40_000,  # 30k mixture + 10k derived binary, headroom
    },
    "TIGER-Lab/MMLU-Pro": {"declared_rows": 12_102, "split": "test", "max_rows": 12_102},
    "allenai/ai2_arc": {"declared_rows": 7_787, "split": "train", "max_rows": 7_787},
    "Rowan/hellaswag": {"declared_rows": 59_950, "split": "train", "max_rows": 59_950},
    "Anthropic/hh-rlhf": {"declared_rows": 169_352, "split": "train", "max_rows": 23_000},
    "stanfordnlp/SHP": {"declared_rows": 385_000, "split": "train", "max_rows": 30_000},
    "nvidia/HelpSteer2": {"declared_rows": 21_362, "split": "train", "max_rows": 21_362},
    "openbmb/UltraFeedback": {
        "declared_rows": 63_967,
        "split": "binarized_prefs_train",
        "config": "binarized",
        "max_rows": 63_967,
    },
    "truthfulqa/truthful_qa": {
        "declared_rows": 1_634,
        "split": "validation",
        "max_rows": 1_634,
    },
    "nvidia/Aegis-AI-Content-Safety-Dataset-2.0": {
        "declared_rows": 33_416,
        "split": "train",
        "max_rows": 18_000,
    },
}

#: Share-alike or otherwise review-flagged licenses that quarantine even when
#: declared (docs/dataset-strategy.md §2 admission rule).
QUARANTINE_LICENSES = frozenset({"cc-by-sa-4.0", "cc-by-nc-4.0", "cc-by-nc-sa-4.0"})


def hub_downloader() -> Downloader:
    """Return the live streaming downloader (imports ``datasets`` lazily)."""

    def _download(source: str, *, split: str | None = None) -> Iterator[Mapping[str, Any]]:
        import datasets  # noqa: PLC0415 - lazy: never imported on the test path

        registry_entry = _SOURCE_REGISTRY.get(source, {})
        kwargs: dict[str, Any] = {"streaming": True, "split": split or registry_entry.get("split")}
        if registry_entry.get("config"):
            kwargs["name"] = registry_entry["config"]
        dataset = datasets.load_dataset(source, **kwargs)
        max_rows = registry_entry.get("max_rows")
        for index, row in enumerate(dataset):
            if max_rows is not None and index >= max_rows:
                break
            yield dict(row)

    return _download


def hub_metadata_for(source: str) -> Any:
    """Fetch one dataset's Hub metadata object (lazy ``HfApi`` import)."""
    from huggingface_hub import HfApi

    return HfApi().dataset_info(source)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Hub dataset id, e.g. cais/mmlu")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-bytes", type=int, default=MAX_CORPUS_BYTES)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="ignore any existing manifest and re-acquire (e.g. after license review)",
    )
    args = parser.parse_args(argv)

    source = args.source
    entry = _SOURCE_REGISTRY.get(source, {})
    try:
        metadata = hub_metadata_for(source)
    except Exception as exc:
        print(f"error: cannot fetch Hub metadata for {source}: {exc}", file=sys.stderr)
        return 2

    out_dir = args.output_root / source.replace("/", "__")
    try:
        manifest: SourceManifest = acquire_source(
            source,
            output_dir=out_dir,
            declared_rows=entry.get("declared_rows"),
            hub_metadata=metadata,
            quarantine_licenses=QUARANTINE_LICENSES,
            split=args.split or entry.get("split"),
            max_bytes=args.max_bytes,
            downloader=hub_downloader(),
            refresh=args.refresh,
        )
    except Exception as exc:  # acquisition is fail-loudly; report and stop
        print(f"error: acquisition of {source} failed: {exc}", file=sys.stderr)
        return 3

    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return 0 if manifest.status != STATUS_QUARANTINE else 1


if __name__ == "__main__":
    sys.exit(main())
