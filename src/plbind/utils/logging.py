import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    fmt: str = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
) -> logging.Logger:
    """Configure root logger with a stream handler and optional file handler.

    Call once from a pipeline entry script. All modules obtain their logger via
    ``logging.getLogger(__name__)`` — no per-module configuration needed.

    Args:
        log_dir: If provided, a ``run.log`` file is written to this directory.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        fmt: Log message format.

    Returns:
        The configured root logger.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Stream handler (stdout)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        root.addHandler(sh)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "run.log")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Silence noisy third-party loggers
    for noisy in ("transformers", "urllib3", "filelock", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root
