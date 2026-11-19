"""Config resolution, validation and run manifests for RAPPORT.

Public surface: ``load_config`` (YAML + RAPPORT_* overrides, fail-fast on
unknown keys), ``ResolvedConfig``/``RunManifest``, ``build_run_manifest``/
``write_run_manifest`` and ``set_seed``.
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
    ManifestWriteError,
    RapportError,
)
from .loader import ResolvedConfig, load_config
from .manifest import (
    RunManifest,
    build_run_manifest,
    compute_config_hash,
    read_git_state,
    read_versions,
    write_run_manifest,
)
from .schema import SCHEMA
from .seeding import set_seed

__all__ = [
    "SCHEMA",
    "ConfigEnvError",
    "ConfigError",
    "ConfigFileUnreadableError",
    "ConfigTypeMismatchError",
    "ConfigUnknownKeyError",
    "ConfigValueError",
    "ConfigYamlError",
    "DependencyMissingError",
    "ManifestError",
    "ManifestWriteError",
    "RapportError",
    "ResolvedConfig",
    "RunManifest",
    "build_run_manifest",
    "compute_config_hash",
    "load_config",
    "read_git_state",
    "read_versions",
    "set_seed",
    "write_run_manifest",
]

__version__ = "0.1.0"
