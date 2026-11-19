# DEC-TASK-02-1 — Package layout: flat src-root modules + `src/rapport/` version carrier

**Context:** acceptance demands `pip install -e .` and `import rapport; rapport.__version__` prints a version, while the architecture contract fixes `source_root: src` with feature modules at `src/<module>/` (mirrored under `tests/unit/`). A `packages.find where=["src"]` install exposes the flat modules (`config`, `model`, ...) as top-level packages — not `rapport`.

**Decision:** keep feature modules flat under `src/` (floor unchanged); add exactly one file `src/rapport/__init__.py` declaring `__version__ = "0.1.0"` so `import rapport` works. No re-exports yet — `rapport/__init__.py` is the version carrier and the future integration point, nothing more. `src/__init__.py` is a docstring-only source-root marker.

**Consequences:** two kinds of import paths coexist (`import rapport` for the distribution, flat `src/<module>` for modules per the architecture contract). Acceptable: the floor gates the mirrored tree, and `import rapport` is the packaging contract. CI and mypy treat `src/` uniformly.

**Alternatives rejected:** mapping `package-dir = {"" = "src"}` alone (fails `import rapport`); phony namespace shim via `src/rapport.py` (module/package collision risk); re-exporting feature modules from `rapport/__init__` (would fake API maturity and create import cycles before the modules exist).
