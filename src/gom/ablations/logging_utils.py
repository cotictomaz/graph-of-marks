"""
Centralized logging for the ablation pipeline.

This module exposes a single dedicated logger (``gom.ablations``) that writes a
human-readable trace of an ablation run to a file under ``{base_dir}/logs/``.
The goal is that reading one log file gives a clear, chronological picture of an
experiment: the full image-preprocessor configuration, the VLM models used and
extra settings, then how the run evolves (dataset built, unique images,
preprocessing progress, VLM inference progress and results).

Design notes
------------
- The logger uses ``propagate = False`` and installs *its own* ``FileHandler``.
  It therefore records only what this ablation pipeline explicitly logs and does
  **not** capture the internal GoM pipeline's own logging/stdout. This keeps the
  log focused on the things that surface at the ablation level.
- Helpers are defensive: a failure while serializing a config must never break
  the experiment, so serialization is wrapped in ``try/except``.
- Setup is idempotent: calling :func:`setup_logging` more than once will not
  attach duplicate handlers.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

_LOGGER_NAME = "gom.ablations"

# Width used for the ASCII section separators in the log file.
_SEP_WIDTH = 70


def setup_logging(base_dir: str, run_name: Optional[str] = None) -> logging.Logger:
    """
    Configure and return the shared ``gom.ablations`` logger.

    A timestamped log file is created under ``{base_dir}/logs/``. Calling this
    function repeatedly is safe: if the logger already has a file handler it is
    reused as-is (no duplicate handlers, no second file).

    Parameters:
        base_dir: Root directory for ablation outputs; logs go to ``{base_dir}/logs``.
        run_name: Optional label included in the log filename for easier lookup.

    Returns:
        The configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    # Do not propagate to the root logger: keeps internal-pipeline / third-party
    # log records out of our file, and keeps our records out of theirs.
    logger.propagate = False

    # Idempotent: if a FileHandler is already attached, don't add another.
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return logger

    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{run_name}" if run_name else ""
    log_path = os.path.join(log_dir, f"ablation{suffix}_{ts}.log")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    logger.info("=" * _SEP_WIDTH)
    logger.info("Ablation pipeline log started")
    logger.info("Log file: %s", log_path)
    logger.info("=" * _SEP_WIDTH)
    return logger


def get_logger() -> logging.Logger:
    """
    Return the shared ``gom.ablations`` logger.

    Safe to call before :func:`setup_logging`; until setup runs the logger simply
    has no file handler and messages are discarded (there is no console handler),
    so callers never need to guard their logging calls.
    """
    return logging.getLogger(_LOGGER_NAME)


def log_section(title: str, logger: Optional[logging.Logger] = None) -> None:
    """Write a visually separated section header to the log."""
    logger = logger or get_logger()
    logger.info("")
    logger.info("=" * _SEP_WIDTH)
    logger.info(title)
    logger.info("=" * _SEP_WIDTH)


def log_key_values(
    title: str,
    values: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Log a dictionary of settings as aligned ``key: value`` lines under a title.

    Used for global run settings, per-experiment settings, and config overrides.
    """
    logger = logger or get_logger()
    logger.info("%s:", title)
    if not values:
        logger.info("    (none)")
        return
    width = max((len(str(k)) for k in values), default=0)
    for key in values:
        logger.info("    %-*s : %s", width, key, values[key])


def log_models(
    models_list: Any,
    backend: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log the VLM models that will be evaluated and the serving backend."""
    logger = logger or get_logger()
    models = list(models_list) if models_list else []
    logger.info("VLM models (%d) via backend '%s':", len(models), backend)
    if not models:
        logger.info("    (none)")
        return
    for entry in models:
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get("model") or "<unnamed>"
            fp8 = entry.get("fp8", entry.get("quantize_fp8", False))
            logger.info("    - %s%s", name, "  [fp8]" if fp8 else "")
        else:
            logger.info("    - %s", entry)


def log_preprocessor_config(
    preproc_obj: Any,
    title: str = "Image preprocessor configuration",
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Log the full configuration of the image preprocessor.

    Reads ``preproc_obj.cfg`` (the :class:`PreprocessorConfig` dataclass) and
    dumps every field as a sorted ``key: value`` list. Any failure here is
    swallowed and noted, so config logging can never break a run.
    """
    logger = logger or get_logger()
    if preproc_obj is None:
        logger.info("%s: preprocessor not initialized (skipping).", title)
        return
    try:
        cfg = getattr(preproc_obj, "cfg", None)
        if cfg is None:
            logger.info("%s: no 'cfg' attribute found on preprocessor.", title)
            return
        if dataclasses.is_dataclass(cfg):
            fields = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
        else:
            fields = dict(vars(cfg))
    except Exception as exc:  # pragma: no cover - defensive only
        logger.warning("%s: could not serialize config (%s).", title, exc)
        return

    logger.info("%s (%d fields):", title, len(fields))
    for key in sorted(fields):
        logger.info("    %-32s : %s", key, fields[key])
