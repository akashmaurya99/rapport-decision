"""Shared helpers for the head tests (imported as a plain top-level module).

Builds joint RAPPORT sequences with the real stand-in tokenizer and the same
layout ``encode_example`` produces (``<task>{family}<goal>{goal}<state>{state}``
followed by ``<opt>…</opt>`` blocks), right-padded to a common width.
"""

from __future__ import annotations

import torch

from src.tokenization.tokenizer import PAD_ID, STATE_ID, RapportTokenizer

_TOKENIZER = RapportTokenizer()


def encode_joint(
    goal: str, state: str, options: list[str], task_family: str = "choice"
) -> list[int]:
    head = _TOKENIZER.encode(f"<task>{task_family}<goal>{goal}")
    ids = [*head, STATE_ID, *_TOKENIZER.encode(state)]
    for option in options:
        ids += _TOKENIZER.encode(f"<opt>{option}</opt>")
    return ids


def make_batch(
    examples: list[tuple[str, str, list[str]]],
    task_ids: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode ``(goal, state, options)`` tuples into a padded batch.

    Returns ``(input_ids, attention_mask, option_mask, task_ids)``. ``option_mask``
    width is the maximum option count in the batch; shorter rows are right-padded.
    """
    encoded = [encode_joint(goal, state, options) for goal, state, options in examples]
    width = max(len(opts) for _, _, opts in examples)
    seq = max(len(ids) for ids in encoded)
    batch_size = len(encoded)

    ids = torch.full((batch_size, seq), PAD_ID, dtype=torch.int64)
    attention = torch.zeros((batch_size, seq), dtype=torch.int64)
    option_mask = torch.zeros((batch_size, width), dtype=torch.int64)
    for row, (encoded_ids, (_, _, options)) in enumerate(zip(encoded, examples)):
        ids[row, : len(encoded_ids)] = torch.tensor(encoded_ids, dtype=torch.int64)
        attention[row, : len(encoded_ids)] = 1
        option_mask[row, : len(options)] = 1
    if task_ids is None:
        task_tensor = torch.zeros(batch_size, dtype=torch.long)
    else:
        task_tensor = torch.tensor(task_ids, dtype=torch.long)
    return ids, attention, option_mask, task_tensor


def example_specs(n_options: int, count: int) -> list[tuple[str, str, list[str]]]:
    base = [
        "alpha beta gamma",
        "delta epsilon zeta",
        "eta theta iota",
        "kappa lambda mu",
        "nu xi omicron",
        "pi rho sigma",
        "tau upsilon phi",
        "chi psi omega",
    ]
    return [
        (f"goal {row}", f"state {row}", base[:n_options]) for row in range(count)
    ]
