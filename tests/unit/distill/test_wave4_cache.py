"""Wave-4 regression tests for the repaired teacher-cache layout."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.core.normalize import normalize_example  # noqa: E402
from src.distill.cache import (  # noqa: E402
    DistillOptions,
    build_soft_label_cache,
    entry_filename,
    template_hash,
)


def _example(index: int):
    return normalize_example(
        {
            "task_family": "choice",
            "goal": f"choose {index}",
            "state": f"context {index}",
            "options": [
                {"text": f"alpha option {index}"},
                {"text": f"beta option {index}"},
            ],
            "source": "fixture/source",
            "license": "MIT",
            "group_id": f"group-{index}",
            "split": "train",
            "chosen_index": 0,
        }
    )


def _teacher(counter: list[int]):
    def teacher(example, prompt):  # noqa: ANN001 - duck-typed teacher
        counter.append(1)
        return [-0.1, -2.0]

    return teacher


def test_two_templates_coexist_without_destroying_each_other(tmp_path: Path) -> None:
    counter: list[int] = []
    options = DistillOptions(cache_dir=str(tmp_path), teacher=_teacher(counter))
    example = _example(0)

    first = build_soft_label_cache([example], "teacher/a", "template A", options)
    second = build_soft_label_cache([example], "teacher/a", "template B", options)
    again = build_soft_label_cache([example], "teacher/a", "template A", options)

    assert first.teacher_forwards == 1
    assert second.teacher_forwards == 1
    assert again.teacher_forwards == 0 and again.hits == 1
    teacher_dir = tmp_path / "teacher_a"
    assert (teacher_dir / entry_filename(example.example_id, template_hash("template A"))).is_file()
    assert (teacher_dir / entry_filename(example.example_id, template_hash("template B"))).is_file()


def test_legacy_entry_is_migrated_instead_of_recomputed(tmp_path: Path) -> None:
    counter: list[int] = []
    options = DistillOptions(cache_dir=str(tmp_path), teacher=_teacher(counter))
    example = _example(1)

    build_soft_label_cache([example], "teacher/b", "template A", options)
    teacher_dir = tmp_path / "teacher_b"
    new_path = teacher_dir / entry_filename(example.example_id, template_hash("template A"))
    legacy_path = teacher_dir / f"{example.example_id}.json"
    new_path.rename(legacy_path)  # simulate the pre-template-keying layout

    result = build_soft_label_cache([example], "teacher/b", "template A", options)
    assert result.teacher_forwards == 0
    assert result.hits == 1
    assert new_path.is_file()
    assert not legacy_path.exists()
