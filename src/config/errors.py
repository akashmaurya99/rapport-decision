"""Typed errors raised by the config module.

Every error carries a stable ``code`` attribute so callers (and tests) can
match on the failure class without parsing message text. The code is included
in ``str(error)`` for log readability.
"""

from __future__ import annotations


class RapportError(Exception):
    """Base class for errors raised by this package."""

    code = "RAPPORT_ERROR"

    def __init__(self, message: str, *, key: str | None = None) -> None:
        super().__init__(f"{self.code}: {message}")
        self.message = message
        self.key = key


class ConfigError(RapportError):
    """Any failure while resolving or validating a run configuration."""

    code = "CONFIG_ERROR"


class ConfigFileUnreadableError(ConfigError):
    """The YAML file could not be read (missing, permission, not a file)."""

    code = "CONFIG_FILE_UNREADABLE"


class ConfigYamlError(ConfigError):
    """The YAML document is unparsable or is not a mapping at the top level."""

    code = "CONFIG_YAML_INVALID"


class ConfigUnknownKeyError(ConfigError):
    """The config document contains a key the schema does not declare."""

    code = "CONFIG_UNKNOWN_KEY"


class ConfigTypeMismatchError(ConfigError):
    """A value has the wrong type for its schema slot."""

    code = "CONFIG_TYPE_MISMATCH"


class ConfigValueError(ConfigError):
    """A value has the right type but is outside the accepted domain."""

    code = "CONFIG_BAD_VALUE"


class ConfigEnvError(ConfigError):
    """A RAPPORT_* environment override could not be coerced to the schema type."""

    code = "CONFIG_BAD_ENV_VALUE"


class DependencyMissingError(RapportError):
    """A package required for seeding/manifests is not importable."""

    code = "DEPENDENCY_MISSING"


class ManifestError(RapportError):
    """The run manifest could not be assembled or written."""

    code = "MANIFEST_ERROR"


class ManifestWriteError(ManifestError):
    """The manifest file could not be written to disk."""

    code = "MANIFEST_WRITE_FAILED"
