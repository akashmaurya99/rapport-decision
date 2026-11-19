# TASK-01 — decisions

1. **Flat schema, no nested sections.** base.yaml exercises only flat keys (seed, device, paths, fp16, loop controls). Nested experiment schemas are explicitly out of scope (delivered by TASK-13). One `KeySpec` table in schema.py is the single source of truth for loader + tests.
2. **data_fingerprint is an env-fed placeholder** (default `""`) recorded verbatim; TASK-07 will replace it with the split-manifest sha256. It is a declared schema key so `RAPPORT_DATA_FINGERPRINT` and YAML both work.
3. **Determinism as an enum** (`strict`/`relaxed`) rather than a bool: the manifest must expose the *resolved* mode; `strict` = deterministic algorithms + no cudnn benchmark + CUBLAS_WORKSPACE_CONFIG setdefault; `relaxed` = seeds only.
4. **git_dirty is tri-state** (`bool | None`): `None` when git is unavailable, recorded as-is; a dirty tree never blocks the run (plan edge case).
5. **Seed domain capped at 2**32-1** (numpy legacy domain) so one seed is valid across random/numpy/torch/CUDA.
6. **Config hash = sha256 of canonical YAML** (`yaml.safe_dump(..., sort_keys=True)`) of the resolved dict — matches the "sha256 of resolved YAML" contract literally, and is stable across dict orderings.
7. **Manifest device field** carries name + sm (e.g. `cuda:0 Tesla T4 (sm_75)`) with compute_capability as a separate list-ish string (`7.5`); on CPU, capability is null.
8. **Unknown RAPPORT_* shell vars are ignored** (only declared keys have env mappings); unknown YAML keys fail with CONFIG_UNKNOWN_KEY. Rationale: the shell may legitimately carry unrelated RAPPORT_* vars; the YAML document is the artifact under validation.
