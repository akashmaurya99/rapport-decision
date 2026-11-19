"""Test-local fixtures for the acquisition suite.

Everything here is offline: the network is replaced by an in-memory
``Downloader`` stub. The live Hub downloader in ``src.data.acquire.hub`` is
never imported by this suite.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXED_NOW = datetime(2026, 11, 19, 12, 0, 0, tzinfo=timezone.utc)


def make_rows(count: int, *, prefix: str = "row") -> list[dict[str, Any]]:
    """Deterministic raw rows shaped like streamed Hub records."""
    rows: list[dict[str, Any]] = []
    for index in range(count):
        rows.append(
            {
                "id": f"{prefix}-{index}",
                "goal": f"Choose the better option for goal {index}",
                "state": f"State {index} with some context.",
                "options": [f"option-{index}-a", f"option-{index}-b"],
                "chosen_index": index % 2,
                "unused_column": f"drop-me-{index}",
            }
        )
    return rows


class StubDownloader:
    """In-memory downloader recording every call."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, source: str, *, split: str | None = None):
        self.calls.append((source, split))
        return iter(list(self.rows))


class FakeCardData:
    """Duck-typed stand-in for ``DatasetCardData``."""

    def __init__(self, *, license: str | list[str] | None) -> None:
        self.license = license


class FakeHubMetadata:
    """Duck-typed stand-in for ``huggingface_hub.DatasetInfo``."""

    def __init__(
        self,
        *,
        license: str | list[str] | None = None,
        num_rows: int | None = None,
        declare_license_slot: bool = True,
    ) -> None:
        self.card_data = FakeCardData(license=license) if declare_license_slot else None
        if num_rows is not None:
            self.num_rows = num_rows


@pytest.fixture
def rows():
    return make_rows


@pytest.fixture
def stub_downloader():
    return StubDownloader


@pytest.fixture
def fixed_now():
    return FIXED_NOW


@pytest.fixture
def fake_card():
    return FakeCardData


@pytest.fixture
def fake_hub():
    return FakeHubMetadata
