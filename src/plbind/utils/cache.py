import functools
import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def disk_cache(cache_dir: Path, suffix: str = ".pkl") -> Callable:
    """Decorator that caches a function's return value to disk.

    The cache key is built from the function name + all positional and keyword
    arguments. Useful for expensive API calls and model inference that should
    only run once per unique input.

    Args:
        cache_dir: Directory where cache files are stored.
        suffix: File extension for cache files.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Build a stable cache key from function name + arguments
            key_source = f"{func.__module__}.{func.__qualname__}:{args!r}:{sorted(kwargs.items())!r}"
            key_hash = hashlib.sha256(key_source.encode()).hexdigest()[:16]
            cache_path = cache_dir / f"{func.__name__}_{key_hash}{suffix}"

            if cache_path.exists():
                logger.debug("Cache hit: %s", cache_path.name)
                with cache_path.open("rb") as f:
                    return pickle.load(f)

            logger.debug("Cache miss: %s — computing...", cache_path.name)
            result = func(*args, **kwargs)
            with cache_path.open("wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            return result

        # Expose a method to clear cached results for this function
        def clear_cache() -> int:
            removed = 0
            if cache_dir.exists():
                for p in cache_dir.glob(f"{func.__name__}_*{suffix}"):
                    p.unlink()
                    removed += 1
            return removed

        wrapper.clear_cache = clear_cache  # type: ignore[attr-defined]
        return wrapper

    return decorator


def numpy_cache(cache_dir: Path) -> Callable:
    """Like disk_cache but stores numpy arrays as .npy files (faster I/O for large arrays)."""
    import numpy as np

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            cache_dir.mkdir(parents=True, exist_ok=True)
            key_source = f"{func.__module__}.{func.__qualname__}:{args!r}:{sorted(kwargs.items())!r}"
            key_hash = hashlib.sha256(key_source.encode()).hexdigest()[:16]
            cache_path = cache_dir / f"{func.__name__}_{key_hash}.npy"

            if cache_path.exists():
                logger.debug("NumPy cache hit: %s", cache_path.name)
                return np.load(cache_path, allow_pickle=False)

            logger.debug("NumPy cache miss: %s — computing...", cache_path.name)
            result = func(*args, **kwargs)
            np.save(cache_path, result)
            return result

        return wrapper

    return decorator
