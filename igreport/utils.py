"""Utility functions for exception reporting."""

import sys
from typing import Any, Dict
from types import (
    ModuleType,
    FunctionType,
    BuiltinFunctionType,
    MethodType,
    BuiltinMethodType,
    CodeType,
    FrameType,
    TracebackType,
)


def _is_excludable_local(name: str, value: Any, frame: FrameType) -> bool:
    """
    Decide whether a local variable should be excluded from the report.

    Excludes:
      - dunder names
      - imported modules
      - functions/methods (builtin or user)
      - classes/metaclasses (incl. builtins like 'int', 'list', etc.)
      - code/frames/tracebacks
      - anything that is exactly the same object as a global with same name
        (typical for imported symbols re-exposed in locals)
    """
    # dunder / sentinel-y names
    if name.startswith("__") and name.endswith("__"):
        return True

    # globals rebound into locals (common for imported symbols)
    # But don't exclude simple values that might be interned (like small integers)
    try:
        if frame and name in frame.f_globals and frame.f_globals[name] is value:
            # Only exclude if it's not a simple primitive value that could be interned
            if not isinstance(value, (int, float, str, bool, type(None))):
                return True
            # For primitives, check if the frame is actually at module level
            # (where globals and locals would legitimately be the same)
            if frame.f_code.co_name == "<module>":
                # At module level, globals and locals are the same dict, so this is expected
                return False
            # In a function/method, if it's a primitive that matches a global,
            # it's likely a local variable, not an imported symbol
            return False
    except Exception:
        pass

    # excludable runtime/container objects
    if isinstance(value, (ModuleType,)):
        return True
    if isinstance(
        value, (FunctionType, BuiltinFunctionType, MethodType, BuiltinMethodType)
    ):
        return True
    if isinstance(value, (type,)):  # classes (incl. builtins) & metaclasses
        return True
    if isinstance(value, (CodeType, FrameType, TracebackType)):
        return True

    # Many "callable" objects are user data holders with __call__; don't blanket-drop callables.
    return False


def _filter_locals_for_report(frame: FrameType) -> dict[str, object]:
    """
    Return a dict of locals with imported/runtime objects removed.
    """
    safe = {}
    locs = frame.f_locals or {}
    for k, v in locs.items():
        if not _is_excludable_local(k, v, frame):
            safe[k] = v
    return safe


def _format_value(value: Any, indent: int = 0) -> str:
    """Format a serialized value for display."""
    if isinstance(value, dict):
        if "__type__" in value:
            # Special serialized object
            obj_type = value["__type__"]
            if obj_type == "bytes":
                return f"<bytes: len={value.get('len', '?')}, preview={value.get('preview', '...')}>"
            elif "repr" in value:
                return f"<{obj_type}: {value['repr']}>"
            elif obj_type in ("set", "frozenset") and "values" in value:
                items = value["values"]
                if len(items) <= 3:
                    formatted_items = ", ".join(str(item) for item in items)
                    return f"{{{formatted_items}}}"
                else:
                    preview = ", ".join(str(item) for item in items[:3])
                    return f"{{{preview}, ...({len(items)} items)}}"
            else:
                return f"<{obj_type}>"
        else:
            # Regular dict
            if not value:
                return "{}"
            elif len(value) == 1 and not any(
                isinstance(v, (dict, list)) for v in value.values()
            ):
                # Simple single-item dict
                k, v = next(iter(value.items()))
                return f"{{{k!r}: {_format_value(v)}}}"
            elif len(value) <= 3 and not any(
                isinstance(v, (dict, list)) for v in value.values()
            ):
                # Small simple dict
                items = ", ".join(
                    f"{k!r}: {_format_value(v)}" for k, v in value.items()
                )
                return f"{{{items}}}"
            else:
                # Complex dict - show summary
                return f"{{...{len(value)} items...}}"

    elif isinstance(value, list):
        if not value:
            return "[]"
        elif len(value) <= 3 and not any(isinstance(v, (dict, list)) for v in value):
            # Simple short list
            items = ", ".join(_format_value(item) for item in value)
            return f"[{items}]"
        else:
            # Long or complex list
            return f"[...{len(value)} items...]"

    elif isinstance(value, str):
        if len(value) > 100:
            return f"{value[:97]!r}..."
        return repr(value)

    else:
        # Primitive or other
        return str(value)