"""
igreport - A library for structured exception reporting and formatting.

This library provides functionality to create detailed, JSON-serializable
exception reports similar to what Sentry provides, along with human-readable
formatting capabilities.

Example usage:
    >>> import igreport
    >>> try:
    ...     raise ValueError("Something went wrong")
    ... except Exception as e:
    ...     report = igreport.create_exception_report(e)
    ...     formatted = igreport.format_exception_report(report)
    ...     print(formatted)
"""

from .constants import REDACT_DEFAULT_KEYS
from .models import (
    Mechanism,
    FrameData,
    ExceptionBlock,
    ThreadInfo,
    OSContext,
    RuntimeContext,
    ProcessContext,
    Contexts,
    ExceptionReport,
)
from .core import create_exception_report
from .formatter import format_exception_report
from .utils import (
    _is_excludable_local,
    _filter_locals_for_report,
    _format_value,
)

__version__ = "0.1.0"
__author__ = "igreport"
__email__ = ""
__description__ = "A library for structured exception reporting and formatting"

# Main API exports
__all__ = [
    # Core functionality
    "create_exception_report",
    "format_exception_report",
    
    # Data models
    "ExceptionReport",
    "ExceptionBlock",
    "FrameData",
    "Mechanism",
    "ThreadInfo",
    "OSContext",
    "RuntimeContext",
    "ProcessContext",
    "Contexts",
    
    # Constants
    "REDACT_DEFAULT_KEYS",
    
    # Utilities (primarily for testing and advanced usage)
    "_is_excludable_local",
    "_filter_locals_for_report",
    "_format_value",
    
    # Version info
    "__version__",
]


# Example usage function
def demo() -> None:
    """
    Demonstrate the library functionality with a sample exception.
    """
    import json

    def boom(n: int) -> float:
        return 10 / n  # ZeroDivisionError when n == 0

    try:
        try:
            v = 0
            boom(v)
        except Exception as inner:
            raise ValueError("Wrapped failure") from inner
    except Exception as e:
        report = create_exception_report(
            e,
            context={"request_id": "abc123", "user_id": 42, "note": "demo"},
            tags={"service": "billing", "region": "eu-central"},
        )

        # Show both JSON and formatted outputs
        print("=== JSON REPORT ===")
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))

        print("\n\n=== FORMATTED REPORT ===")
        formatted = format_exception_report(report)
        print(formatted)


if __name__ == "__main__":
    demo()