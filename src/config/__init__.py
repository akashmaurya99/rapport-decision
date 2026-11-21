"""Config resolution, validation and run manifests for RAPPORT.

Public surface: ``load_experiment_config`` (the validated experiment boundary:
base defaults < experiment YAML < RAPPORT_* overrides, with ``.to_resolved()``,
``.value()`` and ``.document``), ``load_config`` (legacy flat base YAML),
``ResolvedConfig``/``RunManifest``, ``build_run_manifest``/``write_run_manifest``
and ``set_seed``.
"""

from .errors import (
    ConfigEnvError,
    ConfigError,
    ConfigFileUnreadableError,
    ConfigTypeMismatchError,
    ConfigUnknownKeyError,
    ConfigValueError,
    ConfigYamlError,
    DependencyMissingError,
    ManifestError,
    ManifestRunIdError,
    ManifestWriteError,
    RapportError,
)
from .loader import ExperimentConfig, ResolvedConfig, load_config, load_experiment_config
from .manifest import (
    RunManifest,
    build_run_manifest,
    compute_config_hash,
    read_dependency_lock_digest,
    read_git_state,
    read_versions,
    validate_run_id,
    write_run_manifest,
)
from .schema import BASE_DEFAULTS, EXPERIMENT_SCHEMA, SCHEMA, SHARED_FIELDS
from .seeding import set_seed

__all__ = [
    "BASE_DEFAULTS",
    "EXPERIMENT_SCHEMA",
    "SCHEMA",
    "SHARED_FIELDS",
    "ConfigEnvError",
    "ConfigError",
    "ConfigFileUnreadableError",
    "ConfigTypeMismatchError",
    "ConfigUnknownKeyError",
    "ConfigValueError",
    "ConfigYamlError",
    "DependencyMissingError",
    "ExperimentConfig",
    "ManifestError",
    "ManifestRunIdError",
    "ManifestWriteError",
    "RapportError",
    "ResolvedConfig",
    "RunManifest",
    "build_run_manifest",
    "compute_config_hash",
    "load_config",
    "read_dependency_lock_digest",
    "load_experiment_config",
    "read_git_state",
    "read_versions",
    "set_seed",
    "validate_run_id",
    "write_run_manifest",
]

__version__ = "0.1.0"
