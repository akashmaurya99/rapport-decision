"""FR-TOK-02 (bucketing half): at most three padded batch shapes.

No 10,000-example "starter fixture" ships in this repository (the data
fixtures present are two-example files under tests/unit/data/core), so this
module synthesizes one deterministically from a fixed seed — same texts every
run, ~60% short / 25% medium / 15% long states, generating exactly what a
starter fixture would exercise. If a real starter fixture lands in a future
wave, it can replace ``synthetic_starter_fixture`` in the scale test.
"""

from __future__ import annotations

import random

import pytest

from src.data.core.normalize import normalize_example
from src.data.core.schema import DecisionExample
from src.tokenization import (
    PAD_ID,
    PADDING_POLICY,
    TRUNCATION_MANIFEST_KEY,
    TRUNCATION_POLICY,
    bucket_examples,
    encode_example,
    reset_truncation_counts,
    truncation_counts,
)
from src.tokenization import errors as tok_errors

WORD_POOL = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega region primary replica "
    "budget instance failover latency throughput consistency shard quorum "
    "migration workload constraint capacity throughput region primary quorum"
).split()


def synthetic_starter_fixture(seed: int = 20261119, count: int = 10_000) -> list[DecisionExample]:
    rng = random.Random(seed)
    examples: list[DecisionExample] = []
    for index in range(count):
        regime = index % 20
        if regime < 12:  # ~60% short: well inside the 128 bucket
            state_words = rng.randint(2, 6)
        elif regime < 17:  # ~25% medium: lands in the 256 bucket
            state_words = rng.randint(22, 30)
        else:  # ~15% long: overflows the 512 budget and is truncated
            state_words = rng.randint(180, 260)
        state = " ".join(rng.choice(WORD_POOL) for _ in range(state_words))
        goal = " ".join(rng.choice(WORD_POOL) for _ in range(rng.randint(3, 8)))
        texts: list[str] = []
        for _ in range(rng.choice([2, 3, 4])):
            for _attempt in range(16):
                text = " ".join(rng.choice(WORD_POOL) for _ in range(rng.randint(2, 5)))
                if text not in texts:
                    texts.append(text)
                    break
            else:
                texts.append(f"unique tail anchor {len(texts)} {index}")
        option_texts = tuple(texts)
        examples.append(
            normalize_example(
                {
                    "task_family": "choice",
                    "goal": goal,
                    "state": state,
                    "options": [{"text": text, "label": None} for text in option_texts],
                    "source": f"fixture/starter-{index % 3}",
                    "license": "CC-BY-4.0",
                    "group_id": f"starter-{index}",
                    "split": "train",
                    "chosen_index": 0,
                }
            )
        )
    return examples


def make_example(state_words: int, source: str = "fixture/bucket") -> DecisionExample:
    state = " ".join(WORD_POOL[i % len(WORD_POOL)] for i in range(state_words))
    return normalize_example(
        {
            "task_family": "choice",
            "goal": "pick the cheapest migration path",
            "state": state,
            "options": [
                {"text": "lift and shift both regions", "label": None},
                {"text": "consolidate to one region with read replicas", "label": None},
            ],
            "source": source,
            "license": "CC-BY-4.0",
            "group_id": f"bucket-{state_words}-{source}",
            "split": "train",
            "chosen_index": 0,
        }
    )


def setup_function() -> None:
    reset_truncation_counts()


def test_short_examples_pad_to_the_small_shape() -> None:
    examples = [make_example(state_words=3, source=f"fixture/s-{i}") for i in range(5)]
    batches = bucket_examples(examples, batch_size=2)
    assert [b.padded_length for b in batches] == [128, 128, 128]
    assert [b.batch_size for b in batches] == [2, 2, 1]
    for batch in batches:
        assert batch.bucket == "short"
        assert batch.padding_policy == PADDING_POLICY
        assert batch.truncation_policy == TRUNCATION_POLICY


def test_right_pad_policy_mask_and_pad_ids() -> None:
    examples = [make_example(state_words=3, source=f"fixture/mask-{i}") for i in range(2)]
    (batch,) = bucket_examples(examples, batch_size=2)
    for row_ids, row_mask in zip(batch.input_ids, batch.attention_mask):
        assert len(row_ids) == batch.padded_length == len(row_mask)
        seen_zero = False
        for token_id, mask in zip(row_ids, row_mask):
            if mask == 0:
                seen_zero = True
                assert token_id == PAD_ID
            else:
                assert not seen_zero  # no 1 after a 0: strictly right-padded


def test_bucket_assignment_follows_length_edges() -> None:
    short = make_example(state_words=2, source="fixture/edge-short")
    medium = make_example(state_words=28, source="fixture/edge-medium")
    long_ = make_example(state_words=250, source="fixture/edge-long")
    batches = bucket_examples([short, medium, long_], batch_size=1)
    assert [(b.bucket, b.padded_length) for b in batches] == [
        ("short", 128),
        ("medium", 256),
        ("long", 512),
    ]


def test_batches_are_deterministic_across_calls() -> None:
    examples = [
        make_example(state_words=(i * 7) % 40 + 2, source=f"fixture/det-{i}") for i in range(30)
    ]
    first = bucket_examples(examples, batch_size=8)
    second = bucket_examples(examples, batch_size=8)
    assert [(b.bucket, b.padded_length, b.batch_size) for b in first] == [
        (b.bucket, b.padded_length, b.batch_size) for b in second
    ]


def test_decision_examples_are_encoded_and_truncations_counted() -> None:
    long_example = make_example(state_words=260, source="fixture/count-long")
    short_example = make_example(state_words=2, source="fixture/count-short")
    bucket_examples([long_example, short_example])
    assert truncation_counts() == {"fixture/count-long": 1}


def test_truncation_counts_are_isolated_per_caller_manifest() -> None:
    reset_truncation_counts()
    manifest_a: dict = {}
    manifest_b: dict = {}

    bucket_examples([make_example(state_words=260, source="fixture/iso-a")], manifest=manifest_a)
    bucket_examples(
        [
            make_example(state_words=260, source="fixture/iso-b"),
            make_example(state_words=260, source="fixture/iso-b"),
        ],
        manifest=manifest_b,
    )

    assert manifest_a[TRUNCATION_MANIFEST_KEY] == {"fixture/iso-a": 1}
    assert manifest_b[TRUNCATION_MANIFEST_KEY] == {"fixture/iso-b": 2}
    # The legacy global counter was never touched by either caller.
    assert truncation_counts() == {}


def test_more_than_three_bucket_widths_are_refused() -> None:
    example = make_example(state_words=2, source="fixture/refuse")
    with pytest.raises(tok_errors.BudgetTooSmallError):
        bucket_examples([example], bucket_widths=(64, 128, 256, 512))


def test_example_larger_than_the_largest_bucket_is_refused() -> None:
    example = make_example(state_words=250, source="fixture/overflow")
    encoded = encode_example(example)  # 512 tokens, truncated
    reset_truncation_counts()
    with pytest.raises(tok_errors.BucketOverflowError):
        bucket_examples([encoded], bucket_widths=(64, 128, 256))


def test_budget_derives_the_bucket_widths() -> None:
    examples = [make_example(state_words=2, source=f"fixture/w-{i}") for i in range(4)]
    batches = bucket_examples(examples, budget=256, batch_size=4)
    assert {b.padded_length for b in batches} <= {64, 128, 256}
    assert len({b.padded_length for b in batches}) <= 3


def test_at_most_three_distinct_batch_shapes_across_10000_examples() -> None:
    reset_truncation_counts()
    examples = synthetic_starter_fixture(count=10_000)
    batches = bucket_examples(examples, batch_size=32)

    shapes = {batch.padded_length for batch in batches}
    assert len(shapes) <= 3
    assert shapes <= {128, 256, 512}

    total_rows = sum(b.batch_size for b in batches)
    assert total_rows == 10_000

    # Every bucket is represented in this fixture (short/medium/long regimes).
    assert shapes == {128, 256, 512}
    assert {b.bucket for b in batches} == {"short", "medium", "long"}

    # Batch order is short, then medium, then long.
    order = [b.bucket for b in batches]
    assert order == sorted(order, key={"short": 0, "medium": 1, "long": 2}.__getitem__)

    for batch in batches:
        assert batch.padding_policy == PADDING_POLICY
        assert batch.truncation_policy == TRUNCATION_POLICY
        for row_ids, row_mask in zip(batch.input_ids, batch.attention_mask):
            assert len(row_ids) == len(row_mask) == batch.padded_length
            seen_zero = False
            for token_id, mask in zip(row_ids, row_mask):
                if mask == 0:
                    seen_zero = True
                    assert token_id == PAD_ID
                else:
                    assert not seen_zero
        assert len(batch.examples) == batch.batch_size == len(batch.truncated)
    # The long regime overflowed the budget and was counted, per source.
    assert sum(truncation_counts().values()) == sum(
        1 for batch in batches for flag in batch.truncated if flag
    )
