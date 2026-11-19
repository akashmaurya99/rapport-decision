# TASK-03 notes — DecisionExample contract and normalization

Done: 2026-11-19. Single pass, one wave (wave-2 data wave; siblings TASK-05 model trunk, TASK-17 metrics untouched).

## Files shipped
- `src/data/core/__init__.py` — public surface re-exports (records, errors, normalize/validate, id, serializers, parquet helpers).
- `src/data/core/errors.py` — `DataError` base (mirrors `src/config/errors.py` pattern: stable `code`, `str()` prefixes the code, `field` attribute). Distinct codes for the four contracted rejections: `DATA_EMPTY_GOAL`, `DATA_TOO_FEW_OPTIONS`, `DATA_DUPLICATE_OPTION` (E3), `DATA_CHOSEN_INDEX_OUT_OF_RANGE`. Plus typed codes: `DATA_EMPTY_STATE`, `DATA_MISSING_FIELD`, `DATA_EMPTY_FIELD`, `DATA_TOO_MANY_OPTIONS`, `DATA_EMPTY_OPTION`, `DATA_MISSING_CHOSEN_INDEX`, `DATA_BAD_TASK_FAMILY`, `DATA_BAD_SPLIT`, `DATA_SCORES_LENGTH_MISMATCH`, `DATA_SCORE_OUT_OF_RANGE` (NaN/inf fall out of `0.0 <= s <= 1.0` for free), `DATA_RISK_LABEL_INVALID`, `DATA_BAD_TYPE`, `DATA_UNKNOWN_FIELD`, `DATA_ID_MISMATCH`, `DATA_JSON_INVALID`, `DATA_HTML_STRIP_FAILED`, `DATA_PARQUET_UNAVAILABLE`, `DATA_PARQUET_ERROR`. `validate_example` never raises; `normalize_example` raises the FIRST collected error.
- `src/data/core/schema.py` — `Option` and `DecisionExample` frozen dataclasses (`kw_only=True` for the example so required/optional fields may interleave). `compute_example_id` = sha256 of compact-JSON payload `["rapport:decision-example:v1", NFC(task_family), NFC(goal), NFC(state), sorted(NFC(texts))]` — option permutation leaves the hash unchanged (golden test), explicit domain separator prevents cross-payload collisions. `__post_init__` coerces list→tuple (scores through `float()`), then **recomputes the id and raises `DATA_ID_MISMATCH` on disagreement** — a hash is never trusted from outside. `SERIALIZED_FIELDS`/`REQUIRED_FIELDS`/`OPTIONAL_FIELDS` drive unknown-key rejection.
- `src/data/core/normalize.py` — pipeline per text field: NFC → strip HTML via stdlib `HTMLParser` (`convert_charrefs=True` decodes `&amp;`/`&#39;`/numeric refs; `<script>`/`<style>` bodies dropped; tags removed) → collapse all whitespace (incl. NBSP U+00A0, `\t`, `\n`) to single spaces → NFC again. `normalize_example(raw: Mapping)` fail-fast; `validate_example(raw) -> list[DataError]` collect-all; shared `_analyze` single pass. `example_id` supplied in the record is **verified**, never recorded blindly; absent optionals are never invented (stay `None`, omitted in `to_dict`); unknown keys rejected; option-level extra keys rejected; empty normalized option labels recorded as absent (`None`); option ORDER preserved in the record (only the id sorts); HTML chunks joined with a single space so adjacent tags can never fuse words (`foo<br>bar` → `foo bar`).
- `src/data/core/serialize.py` — `to_dict` (absent optionals omitted, `example_id` included as primary key), `from_dict` (delegates to `normalize_example`: parsing IS validation), `to_json`/`from_json` (UTF-8, `ensure_ascii=False`, compact; typed `DATA_JSON_INVALID` on bad input), `write_json`/`read_json` conveniences.
- `src/data/core/parquet.py` — `write_parquet`/`read_parquet`, one row per example, `list<struct<text:string,label:string>>` options column, nullable optionals, round-trip re-validates (nulls dropped = "not provided"). **pyarrow is in the dev venv (26.0.1) but NOT in project dependencies** (pyproject is TASK-02's), so this module imports pyarrow lazily and raises `DATA_PARQUET_UNAVAILABLE` otherwise; parquet tests `importorskip` pyarrow. Parquet was therefore implemented, not silently skipped — but it is optional by construction.
- `tests/unit/data/core/` — 68 tests (schema/normalize/serialize/parquet) + `fixtures/canonical_example.json` (all fields populated; golden `example_id` `555154b5…f73015a6` hard-coded in the test) + conftest putting the repo root on `sys.path` (mirrors `tests/unit/config`).

## Contract rulings (gaps the contract left open, decided here)
1. **Case sensitivity is strict**: `task_family`/`split` must match exactly (config-loader semantics, `_in(...)`); no silent lowering. Families outside the six → `DATA_BAD_TASK_FAMILY`.
2. **`chosen_index` required for `choice` and `preference`** per the architecture-contract field note → distinct `DATA_MISSING_CHOSEN_INDEX`. Not required for other families; `chosen_index: true` (bool) rejected as `DATA_BAD_TYPE`; explicit JSON `null` for an optional field is a type error, an *absent* key means "not provided".
3. **`state` may be empty only for the `preference` family** per the contract note → `DATA_EMPTY_STATE` otherwise.
4. **Options capped at 8** (`DATA_TOO_MANY_OPTIONS`) and ≥2 (`DATA_TOO_FEW_OPTIONS`); the 8-option bound is the contract's default path.
5. **Duplicate detection runs on normalized texts**, so `Yes` vs `  Yes\n`, NFC vs NFD (`café` vs `cafe\u0301`), HTML/entity-wrapped duplicates, and whitespace variants all collapse to one text → `DATA_DUPLICATE_OPTION` (E3).
6. **The eight risk category names are NOT declared anywhere in the repo** (architecture-spec.md consumes `risk[8]` slots; research.md notes Aegis 2.0 ships 13 categories; PRD/architecture.yaml never enumerate them). Membership is therefore **not validated** — only type/non-emptiness/uniqueness/count ≤ 8 (`DATA_RISK_LABEL_INVALID`). The names were NOT invented. When a future task declares the taxonomy, validation belongs in `_risk_labels` (one membership check) — do not retrofit fabricated names.
7. **`example_id` excludes option labels, chosen_index, scores, etc.** — only `(family, goal, state, sorted option texts)` are hashed, per the contract; permutation of options with distinct labels yields the same id.
8. **Empty state for preference** and **goal-only-script** (`<script>…</script>`) → normalize to empty → rejected by `DATA_EMPTY_STATE`/`DATA_EMPTY_GOAL` respectively. A record whose every field is markup is not a record with content we fabricate.
9. `DecisionExample.__post_init__` coerces `list`→`tuple` (json/parquet parse paths) but performs no normalization; `normalize_example` is the only normalize entry point. Constructing with a stale id raises `DATA_ID_MISMATCH` everywhere, not just in `from_dict`.

## Verification
- `.venv/bin/python -m pytest tests/unit/data/core -q` → **68 passed** (parquet tests ran, not skipped: pyarrow 26.0.1 present).
- `.venv/bin/python -m ruff check src/data/core tests/unit/data/core` → All checks passed.
- `.venv/bin/python -m mypy src/data/core` → Success, no issues (6 files).
- Fixture round-trip: `from_json` → `validate_example == []` → `to_dict` → `from_dict` → dataclass equality + identical golden hash asserted; `to_json` byte-equal across round-trips.
- Adversarial spot checks passed: script-only goal → `DATA_EMPTY_GOAL`; NaN score → `DATA_SCORE_OUT_OF_RANGE`; extra option key → `DATA_UNKNOWN_FIELD`.

## Blockers / honest caveats
- Python here is 3.12.12 (repo targets 3.11); nothing 3.12-specific used (`kw_only=True` dataclasses are 3.10+). CI runs 3.11 per TASK-02.
- pyarrow is not a declared dependency; `write_parquet`/`read_parquet` raise `DATA_PARQUET_UNAVAILABLE` in a bare install, and their tests skip. Orchestrator may want pyarrow pinned (or `parquet` extra) in a future wave if parquet IO becomes load-bearing — pyproject.toml is TASK-02's glob, not this task's.
- pandas was deliberately not used for parquet (one less undeclared dep assumption); pyarrow is used directly.
- `risk_labels` membership unvalidated by design (see ruling 6).
- `src/data/__init__.py` intentionally NOT created — not inside `src/data/core/**` ownership; `src.data` works as a PEP 420 namespace package (tests import `src.data.core`, as `src.config` does via its conftest).

## Out of scope honoured
No adapter logic (TASK-09), no dedup/min-hash (TASK-08), no splits (TASK-10), no tokenizer, no model, no pyproject edits, no sibling waves' files, no git operations.
