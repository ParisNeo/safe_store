# src/safestore/__init__.py
from .store import SafeStore, LogLevel
from ascii_colors import ASCIIColors # Expose for user configuration convenience

__version__ = "0.1.0"

__all__ = ["SafeStore", "ASCIIColors", "LogLevel"]