"""Tests for the split-membership fingerprint."""

from __future__ import annotations

import hashlib
import json

import pytest
from conftest import make_example

from src.data.splits import is_sha256_hex, membership_fingerprint, split

_FINGERPRINT_DOMAIN = "rapport:split-membership:v1"


def _expected_fingerprint(pairs):
    canonical = json.dumps(
        [_FINGERPRINT_DOMAIN, sorted(pairs)], ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class TestMembershipFingerprint:
    def test_matches_manual_sha256_of_canonical_sorted_pairs(self):
        pairs = [("b" * 64, "dev"), ("a" * 64, "train"), ("c" * 64, "test")]
        assert membership_fingerprint(pairs) == _expected_fingerprint(pairs)

    def test_is_order_invariant(self):
        pairs = [("b" * 64, "dev"), ("a" * 64, "train"), ("c" * 64, "test")]
        assert membership_fingerprint(pairs) == membership_fingerprint(list(reversed(pairs)))

    def test_empty_membership_has_a_stable_digest(self):
        assert membership_fingerprint([]) == _expected_fingerprint([])

    def test_membership_change_changes_the_fingerprint(self):
        pairs = [("a" * 64, "train")]
        assert membership_fingerprint(pairs) != membership_fingerprint(
            [("a" * 64, "dev")]
        )

    def test_result_manifest_fingerprint_matches_membership(self):
        examples = [make_example(uid=100 + uid, group_id=f"g-{uid}") for uid in range(6)]
        result = split(examples, seed=13)
        membership = [(e.example_id, e.split) for e in result.examples]
        assert result.manifest.fingerprint == membership_fingerprint(membership)

    def test_fingerprint_is_stable_across_calls(self):
        examples = [make_example(uid=200 + uid, group_id=f"g-{uid}") for uid in range(6)]
        assert split(examples, seed=13).fingerprint == split(examples, seed=13).fingerprint


class TestSha256HexValidation:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("a" * 64, True),
            ("0123456789abcdef" * 4, True),
            ("A" * 64, False),
            ("g" * 64, False),
            ("a" * 63, False),
            ("a" * 65, False),
            ("", False),
            (123, False),
            (None, False),
        ],
    )
    def test_is_sha256_hex(self, value, expected):
        assert is_sha256_hex(value) is expected
