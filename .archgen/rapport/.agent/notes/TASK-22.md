# TASK-22 — The DecisionResult prediction API

## What was built

`src/inference/predictor.py` — the inference boundary for RAPPORT (FR-INFER-01).
`tests/unit/inference/test_predictor.py` — 40 tests, CPU-only, ~1s (a 1M-param
custom `RapportConfig` fixture, `torch.manual_seed(20261119)`, random weights;
a separate module-scoped ultra-variant `RapportConfig.ultra()` build exercises
the config-path branch). The real `RapportModel` / `RapportTokenizer` are wired;
nothing is faked.

## Decisions the contract left open

- **DecisionResult carries numbers only.** Fields: `chosen_index: int | None`
  (`None` iff `abstain`), `scores: tuple[float, ...]` (sigmoid utilities,
  uncalibrated), `probabilities: tuple[float, ...]` (softmax at the calibration
  temperature; sums to 1.0), `confidence: float`, `abstain: bool`,
  `risk: tuple[float, ...]` (8 raw sigmoids, no threshold), `temperature: float`.
  No string anywhere — `test_decision_result_schema_has_no_string_field`
  enumerates `typing.get_type_hints` recursively and fails on any `str`, and
  `test_decision_result_field_set_is_exactly_the_contract` pins the field set.
  `task_family` was deliberately NOT added (it would violate the literal
  "no free-text field" wording); task identity stays implicit to the caller.
- **`Predictor(config_path | RapportModel, tokenizer, calibration_map, *, abstention_policy=None, device='cpu')`.**
  - A `RapportModel` instance is used as-is (single load, `eval()` at
    construction). A `str`/`Path` is a YAML mapping with optional `variant`
    (`ultra`|`compact`|`large`, default `compact`) and optional `weights_path`
    (`state_dict`, `map_location=device`, `weights_only=True`). Unknown keys,
    unknown variants and unreadable files raise `PredictorConfigError`
    (`INFER_BAD_CONFIG`) before any compute; nothing half-loads.
    There is no checkpoint registry in the repo yet, so the config schema is
    declared here and documented in the module docstring; TASK-23/24 can build
    a predictor straight from a config file plus a weights blob.
  - `calibration_map` accepts a `CalibrationMap`, a path to a
    `calibration.json` (loaded via `load_calibration_map`), or `None`.
- **Temperature source.** Bucket key `<family>|<2|3-5|6-8>` via
  `make_bucket_key_for_count`; the bucket's `CalibrationEntry` is applied with
  `CalibrationMap.apply(logits, bucket, strict=False)` — an unknown bucket
  falls back to the pooled entry (crashing prediction on a missing bucket
  would violate E1's spirit). **No map supplied → temperature 1.0**, the
  uncalibrated distribution is reported and `temperature` records the identity
  mapping. Platt buckets record 1.0 (the map contract stores 1.0 as the Platt
  placeholder; the Platt parameters do the calibration).
- **`confidence` = max(calibrated probabilities)** — the same quantity
  `CalibrationMap.apply` reports and the same quantity the abstention policy
  consumes; the raw confidence head stays on `RapportOutput` for
  training/eval. `chosen_index` is the argmax of the calibrated probabilities
  (ties → lowest index); softmax is monotonic so this is the argmax of the
  "calibrated scores" the contract refers to.
- **Abstain** uses the predictor's own `AbstentionPolicy` (default: declared
  `(1, -3, 0)` utilities → θ = 0.75, strictly-below rule), never the policy
  frozen into the calibration map — one injectable boundary for the abstain
  decision. When abstaining, scores/probabilities are still returned and
  `chosen_index` is `None`.
- **E1 codes (typed, no traceback):** `INFER_EMPTY_OPTIONS`,
  `INFER_MISSING_STATE`, plus `INFER_MISSING_GOAL`, `INFER_INVALID_OPTION`,
  `INFER_INVALID_OPTION_COUNT` (1 and 9 refused; the model's
  `InvalidOptionCountError` is also caught at the forward boundary and mapped
  to the same code), `INFER_UNKNOWN_TASK_FAMILY`, `INFER_INVALID_REQUEST`.
  All inherit from `PredictorError`; `PredictorConfigError` covers load-time
  failures. Validation runs before any tensor is built.
- **Joint sequence** is built via `encode_example` on a `DecisionExample`
  whose `example_id` is recomputed with `compute_example_id` (options ride the
  real tokenizer path, 512-token budget, state-from-the-end truncation).
  `source="inference"`, `license="unknown"`, `group_id="inference"`,
  `split="test"` are inert markers, never emitted. Duplicate options are the
  caller's schema concern (`normalize_example`); the predictor only refuses
  blank/non-string options.
- **Risk** is passed through as the 8 raw sigmoid values from the risk head —
  no threshold, per the contract (the ad-hoc veto lives in orchestration).

## Verification

- `.venv/bin/python -m pytest tests/unit/inference/test_predictor.py -q` → 40 passed
- `.venv/bin/python -m ruff check src/inference/predictor.py tests/unit/inference/test_predictor.py` → clean
- `.venv/bin/python -m mypy src/inference/predictor.py` → no issues

Note: with random tiny-model weights the distribution is near-uniform, so
happy-path argmax/field tests inject a zero-threshold policy
(`_never_abstain`); the default θ=0.75 abstain path is covered explicitly.
