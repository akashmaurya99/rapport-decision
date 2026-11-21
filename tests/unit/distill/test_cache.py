"""FR-DISTILL-01: content-addressed soft-label cache behaviour.

Every test uses an injected deterministic stub teacher; no network access,
no live model download.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.distill import (
    DEFAULT_CACHE_ROOT,
    DistillOptions,
    build_soft_label_cache,
    entry_filename,
    index_key,
    sanitize_teacher_id,
    template_hash,
)
from src.distill.cache import CACHE_VERSION


def _three_examples(factory):
    return [
        factory(goal=f"Goal number {i} for the deployment rollout decision")
        for i in range(3)
    ]


def test_template_hash_is_sha256_and_stable() -> None:
    import hashlib

    digest = template_hash("Goal: {goal}\nAnswer:")
    assert digest == hashlib.sha256("Goal: {goal}\nAnswer:".encode("utf-8")).hexdigest()
    assert len(digest) == 64
    assert digest == template_hash("Goal: {goal}\nAnswer:")
    assert template_hash("a") != template_hash("b")


def test_template_hash_rejects_empty() -> None:
    from src.distill import DistillInputError

    with pytest.raises(DistillInputError):
        template_hash("")


def test_sanitize_teacher_id_makes_path_safe() -> None:
    assert sanitize_teacher_id("Qwen/Qwen3-1.7B") == "Qwen_Qwen3-1.7B"
    assert "/" not in sanitize_teacher_id("../../etc/passwd")
    assert sanitize_teacher_id("plain_teacher-1.0") == "plain_teacher-1.0"


def test_first_invocation_calls_teacher_and_writes_cache(tmp_path, example_factory,
                                                        stub_teacher_factory) -> None:
    examples = _three_examples(example_factory)
    stub = stub_teacher_factory([2.0, 0.0, -1.0])
    options = DistillOptions(cache_dir=tmp_path, teacher=stub)

    result = build_soft_label_cache(examples, "teacher/stub-1.7B", "Goal: {goal}\nAnswer:", options)

    assert stub.calls == 3
    assert result.teacher_forwards == 3
    assert result.hits == 0
    assert result.misses == 3
    assert result.cache_hit_rate == 0.0
    teacher_dir = Path(result.cache_dir)
    assert teacher_dir == tmp_path / "teacher_stub-1.7B"
    template = "Goal: {goal}\nAnswer:"
    for example in examples:
        entry_path = teacher_dir / entry_filename(
            example.example_id, template_hash(template)
        )
        payload = json.loads(entry_path.read_text("utf-8"))
        assert payload["version"] == CACHE_VERSION
        assert payload["teacher_id"] == "teacher/stub-1.7B"
        assert payload["template_hash"] == template_hash("Goal: {goal}\nAnswer:")
        assert payload["example_id"] == example.example_id
        assert all(isinstance(v, float) for v in payload["logprobs"])
        assert payload["payload_hash"]
    index = json.loads((teacher_dir / "index.json").read_text("utf-8"))
    assert set(index["entries"]) == {
        index_key(template_hash(template), e.example_id) for e in examples
    }


def test_second_invocation_performs_zero_teacher_forwards(tmp_path, example_factory,
                                                          stub_teacher_factory) -> None:
    examples = _three_examples(example_factory)
    stub = stub_teacher_factory([3.0, 1.0, 0.0])
    options = DistillOptions(cache_dir=tmp_path, teacher=stub)
    template = "Goal: {goal}\n{state}\n{options}\nAnswer:"

    first = build_soft_label_cache(examples, "stub", template, options)
    assert first.teacher_forwards == 3

    second_stub = stub_teacher_factory([3.0, 1.0, 0.0])
    second_options = DistillOptions(cache_dir=tmp_path, teacher=second_stub)
    second = build_soft_label_cache(examples, "stub", template, second_options)

    assert second.teacher_forwards == 0
    assert second_stub.calls == 0
    assert second.hits == 3
    assert second.misses == 0
    assert second.cache_hit_rate == 1.0
    assert second.recomputed == 0
    for example in examples:
        assert second.labels[example.example_id].logprobs == tuple(
            first.labels[example.example_id].logprobs
        )


def test_key_covers_template_hash_and_teacher_id(tmp_path, example_factory,
                                                 stub_teacher_factory) -> None:
    examples = _three_examples(example_factory)
    options = DistillOptions(cache_dir=tmp_path, teacher=stub_teacher_factory())
    build_soft_label_cache(examples, "stub", "template A", options)

    # same teacher, different template -> zero carry-over
    other = DistillOptions(cache_dir=tmp_path, teacher=stub_teacher_factory([9.0, 9.0, 9.0]))
    result = build_soft_label_cache(examples, "stub", "template B", other)
    assert result.teacher_forwards == 3
    assert result.hits == 0

    # same template, different teacher -> separate directory, full miss
    other_teacher = DistillOptions(cache_dir=tmp_path, teacher=stub_teacher_factory())
    result = build_soft_label_cache(examples, "other-teacher", "template A", other_teacher)
    assert result.teacher_forwards == 3
    assert Path(result.cache_dir).name == "other-teacher"


def test_template_a_b_a_keeps_hits_instead_of_thrashing(
    tmp_path, example_factory, stub_teacher_factory
) -> None:
    examples = _three_examples(example_factory)
    template_a = "template A: {goal}\n{state}\n{options}"
    template_b = "template B: {goal}\n{state}\n{options}"

    first = build_soft_label_cache(
        examples, "stub", template_a,
        DistillOptions(cache_dir=tmp_path, teacher=stub_teacher_factory([1.0, 0.0, -1.0])),
    )
    assert first.teacher_forwards == 3

    second = build_soft_label_cache(
        examples, "stub", template_b,
        DistillOptions(cache_dir=tmp_path, teacher=stub_teacher_factory([2.0, 1.0, 0.0])),
    )
    assert second.teacher_forwards == 3
    assert second.hits == 0

    third_stub = stub_teacher_factory([9.0, 9.0, 9.0])
    third = build_soft_label_cache(
        examples, "stub", template_a,
        DistillOptions(cache_dir=tmp_path, teacher=third_stub),
    )
    assert third.teacher_forwards == 0
    assert third_stub.calls == 0
    assert third.hits == 3
    assert third.cache_hit_rate == 1.0
    assert all(
        third.labels[e.example_id].logprobs == first.labels[e.example_id].logprobs
        for e in examples
    )


def test_corrupt_entry_is_recomputed_once_and_new_hash_recorded(
    tmp_path, example_factory, stub_teacher_factory
) -> None:
    examples = _three_examples(example_factory)
    template = "Goal: {goal}\nAnswer:"
    stub = stub_teacher_factory([1.0, 0.5, 0.25])
    options = DistillOptions(cache_dir=tmp_path, teacher=stub)
    first = build_soft_label_cache(examples, "stub", template, options)
    assert first.teacher_forwards == 3

    target = examples[0]
    entry_path = Path(first.cache_dir) / entry_filename(
        target.example_id, template_hash(template)
    )
    payload = json.loads(entry_path.read_text("utf-8"))
    payload["logprobs"][0] = 99.0  # corrupt the payload, keep the old hash
    entry_path.write_text(json.dumps(payload), "utf-8")

    recompute_stub = stub_teacher_factory([1.0, 0.5, 0.25])
    result = build_soft_label_cache(
        examples, "stub", template, DistillOptions(cache_dir=tmp_path, teacher=recompute_stub)
    )
    assert result.teacher_forwards == 1  # exactly the corrupt entry
    assert recompute_stub.calls == 1
    assert result.recomputed == 1
    assert result.hits == 2
    repaired = json.loads(entry_path.read_text("utf-8"))
    assert repaired["logprobs"][0] == pytest.approx(1.0)
    assert repaired["payload_hash"] != payload["payload_hash"]

    # third run: repaired entry is trusted, zero forwards
    third_stub = stub_teacher_factory([1.0, 0.5, 0.25])
    third = build_soft_label_cache(
        examples, "stub", template, DistillOptions(cache_dir=tmp_path, teacher=third_stub)
    )
    assert third.teacher_forwards == 0
    assert third.cache_hit_rate == 1.0


def test_unreadable_cache_file_is_a_miss_not_an_error(tmp_path, example_factory,
                                                       stub_teacher_factory) -> None:
    examples = _three_examples(example_factory)
    template = "Goal: {goal}\nAnswer:"
    options = DistillOptions(cache_dir=tmp_path, teacher=stub_teacher_factory())
    build_soft_label_cache(examples, "stub", template, options)
    target = (
        Path(options.cache_dir)
        / "stub"
        / entry_filename(examples[0].example_id, template_hash(template))
    )
    target.write_bytes(b"\xff\xfe not json")

    stub = stub_teacher_factory()
    result = build_soft_label_cache(
        examples, "stub", template, DistillOptions(cache_dir=tmp_path, teacher=stub)
    )
    assert result.teacher_forwards == 1
    assert result.recomputed == 1


def test_wrong_recorded_key_is_a_miss(tmp_path, example_factory, stub_teacher_factory) -> None:
    examples = _three_examples(example_factory)
    template = "Goal: {goal}\nAnswer:"
    options = DistillOptions(cache_dir=tmp_path, teacher=stub_teacher_factory())
    build_soft_label_cache(examples, "stub", template, options)
    target = (
        Path(options.cache_dir)
        / "stub"
        / entry_filename(examples[0].example_id, template_hash(template))
    )
    payload = json.loads(target.read_text("utf-8"))
    payload["template_hash"] = "0" * 64  # valid JSON, wrong key component
    target.write_text(json.dumps(payload), "utf-8")  # hash now mismatches too

    stub = stub_teacher_factory()
    result = build_soft_label_cache(
        examples, "stub", template, DistillOptions(cache_dir=tmp_path, teacher=stub)
    )
    assert result.teacher_forwards == 1


def test_result_is_json_serializable(tmp_path, example_factory, stub_teacher_factory) -> None:
    examples = _three_examples(example_factory)
    options = DistillOptions(cache_dir=tmp_path, teacher=stub_teacher_factory())
    result = build_soft_label_cache(examples, "stub", "Goal: {goal}\nAnswer:", options)
    encoded = json.dumps(result.to_dict())
    assert encoded
    decoded = json.loads(encoded)
    assert decoded["total"] == 3
    assert decoded["cache_hit_rate"] == 0.0


def test_default_cache_root_documented() -> None:
    assert DEFAULT_CACHE_ROOT == "cache/teacher"


def test_prompt_contains_goal_state_and_lettered_options(example_factory) -> None:
    from src.distill import render_prompt

    example = example_factory(goal="Choose quickly", state="Everything is on fire")
    prompt = render_prompt("Goal: {goal}\n{state}\n{options}\nAnswer:", example)
    assert "Goal: Choose quickly" in prompt
    assert "Everything is on fire" in prompt
    assert "A. Roll out to ten percent" in prompt
    assert "B. Roll out to all traffic" in prompt
    assert prompt.endswith("Answer:")


def test_empty_batch_is_a_clean_noop(tmp_path, stub_teacher_factory) -> None:
    stub = stub_teacher_factory()
    result = build_soft_label_cache(
        [], "stub", "Goal: {goal}\nAnswer:", DistillOptions(cache_dir=tmp_path, teacher=stub)
    )
    assert result.total == 0
    assert result.teacher_forwards == 0
    assert stub.calls == 0
    assert result.cache_hit_rate == 1.0


def test_duplicate_examples_in_one_batch_share_one_forward(tmp_path, example_factory,
                                                           stub_teacher_factory) -> None:
    example = example_factory(goal="The same decision goal about the rollout plan")
    stub = stub_teacher_factory()
    result = build_soft_label_cache(
        [example, example], "stub", "Goal: {goal}\nAnswer:",
        DistillOptions(cache_dir=tmp_path, teacher=stub),
    )
    assert result.teacher_forwards == 1
    assert result.total == 2
    assert result.hits == 0
    assert result.misses == 2
