"""Split-membership fingerprint.

The fingerprint commits to the exact set of (example_id, split) pairs in the
split output. It is order-invariant (membership is sorted before hashing) and
platform-stable (compact JSON, sha256), so two runs over the same input with
the same seed produce the same fingerprint on any machine.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable

# Domain separator prefixed to the hashed payload so a split-membership
# digest can never collide with a digest of some other JSON payload.
_FINGERPRINT_DOMAIN = "rapport:split-membership:v1"

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def is_sha256_hex(value: object) -> bool:
    """True iff ``value`` is a lowercase sha256 hex digest."""
    return isinstance(value, str) and bool(_SHA256_HEX.match(value))


def membership_fingerprint(membership: Iterable[tuple[str, str]]) -> str:
    """Return the sha256 fingerprint of ``(example_id, split)`` membership.

    Pairs are sorted by ``(example_id, split)`` before hashing, so permuting
    the input order never changes the digest. Duplicate pairs are retained
    (multiset semantics): the fingerprint then commits to the multiset, not
    just the set.
    """
    pairs = sorted((str(example_id), str(split)) for example_id, split in membership)
    payload: list[object] = [_FINGERPRINT_DOMAIN, pairs]
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
