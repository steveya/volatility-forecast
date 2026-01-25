"""Deprecated shim for cache backends.

Cache backends now live in alphaforge.data.cache.
"""

from alphaforge.data.cache import CacheBackend, FileCacheBackend, DuckDBCacheBackend

__all__ = ["CacheBackend", "FileCacheBackend", "DuckDBCacheBackend"]
