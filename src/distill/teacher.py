"""The local teacher: protocol, prompt rendering and the live HF path.

Stage C (training-strategy.md §4) uses ``Qwen/Qwen3-1.7B`` as a teacher that
runs **locally inside the runtime**. The only network traffic permitted is the
one-time download performed by Hugging Face on first use; after that the model
and tokenizer are plain local files and every forward pass is offline.

The unit tests never touch this live path: ``build_soft_label_cache`` takes a
teacher *callable* via ``options.teacher`` (a deterministic stub in tests).
The live path exists for the Stage C run and is imported lazily so the rest of
the package works without ``transformers`` installed.

Teacher protocol
----------------
A teacher is any callable

    ``teacher(example: DecisionExample, prompt: str) -> Sequence[float]``

returning exactly one **log-probability per option**, in option order, of the
option string at the answer position. For the live path the log-prob is the
sum of the option completion tokens' log-probs given the rendered prompt
(the standard multiple-choice scoring scheme); calibration and temperature
scaling are applied by the consumer, not here.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from ..data.core import DecisionExample
from .errors import DistillInputError, TeacherUnavailableError

#: Default Stage C answer-position template. ``{goal}``, ``{state}`` and
#: ``{options}`` are substituted; ``{options}`` renders one ``A. text`` line
#: per option. Any other brace is left untouched.
DEFAULT_TEMPLATE = (
    "Goal: {goal}\n"
    "State: {state}\n"
    "Options:\n{options}\n"
    "Answer:"
)

#: Letter labels used to enumerate options in a rendered prompt.
_OPTION_LETTERS = "ABCDEFGH"


@runtime_checkable
class Teacher(Protocol):
    """Callable teacher protocol; see module docstring."""

    def __call__(self, example: DecisionExample, prompt: str) -> Sequence[float]:
        ...


def render_prompt(template: str, example: DecisionExample) -> str:
    """Render ``template`` for ``example``.

    Supported placeholders: ``{goal}``, ``{state}`` and ``{options}``
    (one ``LETTER. text`` line per option). Unknown placeholders are left
    as-is. Raises :class:`DistillInputError` for a non-string template.
    """
    if not isinstance(template, str) or not template:
        raise DistillInputError("template must be a non-empty string", field="template")
    options_block = "\n".join(
        f"{_OPTION_LETTERS[index] if index < len(_OPTION_LETTERS) else index}. {option.text}"
        for index, option in enumerate(example.options)
    )
    # format_map with a default-dict so unknown placeholders survive verbatim.
    class _Safe(dict):  # type: ignore[type-arg]
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(
        _Safe(goal=example.goal, state=example.state, options=options_block)
    )


def load_local_teacher(
    teacher_id: str,
    *,
    device: str = "cpu",
    torch_dtype: str = "auto",
) -> "LocalCausalLMTeacher":
    """Load the teacher as a local causal LM (live path; never used in tests).

    ``transformers`` is imported lazily. The first call may download the model
    from the Hugging Face hub (the *one-time* download the contract allows);
    afterwards the model is served from the local cache with no network
    access. Any failure to load raises :class:`TeacherUnavailableError`
    (E8's fail-to-load limb) instead of a raw transformers exception.
    """
    if not isinstance(teacher_id, str) or not teacher_id.strip():
        raise DistillInputError("teacher_id must be a non-empty string", field="teacher_id")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise TeacherUnavailableError(
            f"transformers/torch are required for the live teacher path; install them "
            f"or inject a teacher callable: {exc}"
        ) from exc

    try:
        tokenizer: Any = AutoTokenizer.from_pretrained(teacher_id)
        model: Any = AutoModelForCausalLM.from_pretrained(teacher_id, torch_dtype=torch_dtype)
        model.to(device)
        model.eval()
    except Exception as exc:  # transformers raises many error types; unify them
        raise TeacherUnavailableError(
            f"could not load teacher {teacher_id!r} locally: {exc}"
        ) from exc
    return LocalCausalLMTeacher(model, tokenizer, device=device, torch_dtype=torch_dtype)


class LocalCausalLMTeacher:
    """Multiple-choice scorer over a locally loaded causal LM.

    Scores each option by summing the log-probs of its token ids at the
    answer position (greedy completion). Runs under ``torch.no_grad`` in
    FP32 on the requested device; performs no network calls once loaded.
    """

    def __init__(self, model: Any, tokenizer: Any, *, device: str = "cpu",
                 torch_dtype: str = "auto") -> None:
        self._model: Any = model
        self._tokenizer: Any = tokenizer
        self.device = device
        self.torch_dtype = torch_dtype

    def __call__(self, example: DecisionExample, prompt: str) -> list[float]:
        import torch  # local import: live path only

        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None

        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        scores: list[float] = []
        with torch.no_grad():
            for option in example.options:
                completion = " " + option.text
                completion_ids = tokenizer(
                    completion, return_tensors="pt", add_special_tokens=False
                ).input_ids.to(self.device)
                if completion_ids.shape[1] == 0:
                    scores.append(float("-inf"))
                    continue
                full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
                logits = model(full_ids).logits
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                start = prompt_ids.shape[1]
                total = 0.0
                for offset in range(completion_ids.shape[1]):
                    position = start + offset
                    token_id = int(full_ids[0, position])
                    # logits at position-1 predict token at position
                    total += float(log_probs[0, position - 1, token_id])
                scores.append(total)
        return scores
