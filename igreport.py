import sys
import os
import uuid
import json
import time
import socket
import platform
import linecache
import threading
from dataclasses import dataclass, asdict
from typing import Any, Optional, Dict, List, Union
from types import TracebackType
from collections.abc import Mapping, Sequence
from types import (
    ModuleType,
    FunctionType,
    BuiltinFunctionType,
    MethodType,
    BuiltinMethodType,
    CodeType,
    FrameType,
)

REDACT_DEFAULT_KEYS = {
    "password",
    "passwd",
    "pwd",
    "secret",
    "api_secret",
    "client_secret",
    "token",
    "access_token",
    "refresh_token",
    "id_token",
    "key",
    "api_key",
    "authorization",
    "auth",
    "cookie",
    "set-cookie",
    "private_key",
}


@dataclass
class Mechanism:
    """Exception mechanism information."""

    type: str
    handled: bool


@dataclass
class FrameData:
    """Stack frame information."""

    abs_path: str
    filename: str
    function: str
    module: Optional[str]
    lineno: int
    colno: Optional[int]
    pre_context: List[str]
    pre_context_lineno: Optional[int]
    context_line: Optional[str]
    post_context: List[str]
    in_app: bool
    vars: Optional[Any] = None  # Can be dict or other serialized types
    vars_error: Optional[str] = None


@dataclass
class ExceptionBlock:
    """Exception information with frames."""

    type: str
    module: Optional[str]
    message: str
    mechanism: Mechanism
    frames: List[Union[FrameData, Dict[str, str]]]  # Dict for truncated frames
    relation: str = "exception"


@dataclass
class ThreadInfo:
    """Thread information."""

    current_thread_id: int
    current_thread_name: str


@dataclass
class OSContext:
    """Operating system context."""

    name: str
    version: str
    release: str
    platform: str


@dataclass
class RuntimeContext:
    """Runtime context."""

    name: str
    version: str
    build: tuple
    implementation: str


@dataclass
class ProcessContext:
    """Process context."""

    pid: int
    argv: List[str]
    executable: str
    cwd: str


@dataclass
class Contexts:
    """Environment contexts."""

    os: Optional[OSContext] = None
    runtime: Optional[RuntimeContext] = None
    process: Optional[ProcessContext] = None


@dataclass
class ExceptionReport:
    """Main exception report structure."""

    event_id: str
    timestamp: str
    level: str
    platform: str
    logger: Optional[str]
    transaction: Optional[str]
    server_name: str
    release: Optional[str]
    environment: Optional[str]
    tags: Dict[str, str]
    extra: Any  # Can be various serialized types
    threads: ThreadInfo
    exceptions: List[ExceptionBlock]
    contexts: Optional[Contexts] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExceptionReport":
        """Create from dictionary."""
        # Handle contexts conversion
        contexts_data = data.get("contexts")
        contexts = None
        if contexts_data:
            contexts = Contexts(
                os=OSContext(**contexts_data["os"]) if "os" in contexts_data else None,
                runtime=RuntimeContext(**contexts_data["runtime"])
                if "runtime" in contexts_data
                else None,
                process=ProcessContext(**contexts_data["process"])
                if "process" in contexts_data
                else None,
            )

        # Handle threads conversion
        threads_data = data["threads"]
        threads = ThreadInfo(**threads_data)

        # Handle exceptions conversion
        exceptions = []
        for exc_data in data["exceptions"]:
            mechanism = Mechanism(**exc_data["mechanism"])

            frames = []
            for frame_data in exc_data["frames"]:
                if "truncated" in frame_data:
                    frames.append(frame_data)  # Keep as dict for truncated frames
                else:
                    frames.append(FrameData(**frame_data))

            exception_block = ExceptionBlock(
                type=exc_data["type"],
                module=exc_data["module"],
                message=exc_data["message"],
                mechanism=mechanism,
                frames=frames,
                relation=exc_data.get("relation", "exception"),
            )
            exceptions.append(exception_block)

        return cls(
            event_id=data["event_id"],
            timestamp=data["timestamp"],
            level=data["level"],
            platform=data["platform"],
            logger=data["logger"],
            transaction=data["transaction"],
            server_name=data["server_name"],
            release=data["release"],
            environment=data["environment"],
            tags=data["tags"],
            extra=data["extra"],
            threads=threads,
            exceptions=exceptions,
            contexts=contexts,
        )


def _is_excludable_local(name: str, value, frame: FrameType) -> bool:
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


def create_exception_report(
    exc: BaseException | None = None,
    /,
    *,
    context: dict | None = None,
    tags: dict | None = None,
    level: str = "error",
    max_frames: int = 200,
    context_lines: int = 3,
    capture_locals: bool = True,
    redact_keys: set[str] | None = None,
    max_str_len: int = 1024,
    max_container_items: int = 50,
    max_depth: int = 3,
    include_env: bool = True,
) -> ExceptionReport:
    """
    Build a structured, JSON-serializable report of the *current* exception
    (or the provided `exc`), similar to Sentry.

    Parameters:
      exc: the exception instance to report. If None, uses sys.exc_info().
      context: arbitrary extra context to attach (must be JSON-serializable or repr-able).
      tags: lightweight labels (small strings) for quick filtering.
      level: log level string, e.g. "error", "fatal", "warning".
      max_frames: safeguard for maximum number of frames per exception in the chain.
      context_lines: how many pre/post source lines to include per frame.
      capture_locals: include local variables per frame (safely serialized + redacted).
      redact_keys: set of lowercase keys to redact; merged with REDACT_DEFAULT_KEYS.
      max_str_len: truncate long strings/reprs to this length.
      max_container_items: cap items per list/dict/set when serializing.
      max_depth: max nested depth for serializer.
      include_env: include process, platform, and thread info.

    Returns:
      An ExceptionReport instance.
    """
    # Resolve exception & traceback
    if exc is None:
        _, exc_obj, _ = sys.exc_info()
        exc = exc_obj
    if exc is None:
        raise RuntimeError(
            "create_exception_report must be called inside an except block or provided an exception."
        )

    redactions = (redact_keys or set()) | REDACT_DEFAULT_KEYS

    # ---- helpers -------------------------------------------------------------

    def _is_primitive(x) -> bool:
        return isinstance(x, (type(None), bool, int, float, str))

    def _safe_str(x) -> str:
        try:
            s = str(x)
        except Exception:
            try:
                s = repr(x)
            except Exception:
                s = f"<unprintable {type(x).__name__}>"
        if len(s) > max_str_len:
            return s[:max_str_len] + f"...(truncated {len(s) - max_str_len} chars)"
        return s

    def _redact_key(k: str) -> bool:
        k_low = str(k).lower()
        # straightforward equality match OR contains sensitive tokens like "password"
        if k_low in redactions:
            return True
        # heuristic: contains sensitive substrings
        substrings = ("password", "secret", "token", "auth", "key", "cookie", "passwd")
        return any(s in k_low for s in substrings)

    def _serialize(obj, depth=0):
        if depth >= max_depth:
            return f"<max_depth:{max_depth} {type(obj).__name__}>"

        if _is_primitive(obj):
            return obj if not isinstance(obj, str) else _safe_str(obj)

        # bytes -> show length + preview
        if isinstance(obj, (bytes, bytearray, memoryview)):
            b = bytes(obj)
            preview = b[:32]
            return {
                "__type__": "bytes",
                "len": len(b),
                "preview": preview.hex(),
            }

        # mappings
        if isinstance(obj, Mapping):
            out = {}
            count = 0
            for k, v in obj.items():
                if count >= max_container_items:
                    out["__truncated__"] = f"{count}+ items"
                    break
                sk = _safe_str(k)
                if _redact_key(sk):
                    out[sk] = "<redacted>"
                else:
                    out[sk] = _serialize(v, depth + 1)
                count += 1
            return out

        # sequences (but not str/bytes handled above)
        if isinstance(obj, Sequence):
            out_list = []
            for i, item in enumerate(obj):
                if i >= max_container_items:
                    out_list.append(f"<truncated:{len(obj) - i} more items>")
                    break
                out_list.append(_serialize(item, depth + 1))
            return out_list

        # sets
        if isinstance(obj, set) | isinstance(obj, frozenset):
            out_list = []
            for i, item in enumerate(obj):
                if i >= max_container_items:
                    out_list.append(f"<truncated:{len(obj) - i} more items>")
                    break
                out_list.append(_serialize(item, depth + 1))
            return {"__type__": type(obj).__name__, "values": out_list}

        # fallback: type + safe repr
        return {
            "__type__": type(obj).__name__,
            "repr": _safe_str(obj),
        }

    def _source_context(abs_path: str, lineno: int, ctx_lines: int):
        if not abs_path or lineno is None:
            return [], None, []
        # Ensure linecache has fresh view
        linecache.checkcache(abs_path)
        start = max(1, lineno - ctx_lines)
        end = lineno + ctx_lines
        pre = []
        for i in range(start, lineno):
            pre.append((i, (linecache.getline(abs_path, i) or "").rstrip("\n")))
        line = (linecache.getline(abs_path, lineno) or "").rstrip("\n")
        post = []
        for i in range(lineno + 1, end + 1):
            post.append((i, (linecache.getline(abs_path, i) or "").rstrip("\n")))
        return pre, line, post

    def _create_frame_data(frame, lineno: int) -> FrameData:
        # frame: types.FrameType
        f_code = frame.f_code
        abs_path = os.path.abspath(f_code.co_filename)
        filename = os.path.basename(abs_path)
        pre, line, post = _source_context(abs_path, lineno, context_lines)

        frame_data = FrameData(
            abs_path=abs_path,
            filename=filename,
            function=f_code.co_name,
            module=frame.f_globals.get("__name__", None),
            lineno=lineno,
            colno=None,  # Python doesn't provide column by default
            pre_context=[s for _, s in pre],
            pre_context_lineno=pre[0][0] if pre else None,
            context_line=line,
            post_context=[s for _, s in post],
            in_app=_guess_in_app(abs_path),
        )

        if capture_locals:
            try:
                filtered = _filter_locals_for_report(frame)
                frame_data.vars = _serialize(filtered, depth=0)
            except Exception as e:
                frame_data.vars_error = _safe_str(e)

        return frame_data

    def _guess_in_app(path: str) -> bool:
        # crude heuristic similar to how many SDKs differentiate framework vs app frames
        # Treat stdlib & site-packages as not in_app
        stdlib = os.__file__.rsplit(os.sep, 2)[0]
        site = next(
            (p for p in sys.path if "site-packages" in p or "dist-packages" in p), None
        )
        if path.startswith(stdlib):
            return False
        if site and path.startswith(site):
            return False
        return True

    def _iter_chain(e: BaseException):
        """
        Yields (exc, relation) following Python's exception chaining rules:
        - primary exception
        - then __cause__ chain (explicit)
        - if no cause and not suppressed, follow __context__ (implicit)
        """
        visited = set()
        current = e
        relation = "exception"
        while current and id(current) not in visited:
            visited.add(id(current))
            yield current, relation

            if current.__cause__ is not None:
                current, relation = current.__cause__, "cause"
                continue
            if not current.__suppress_context__ and current.__context__ is not None:
                current, relation = current.__context__, "context"
                continue
            break

    def _create_exception_block(e: BaseException) -> ExceptionBlock:
        # walk traceback frames (from oldest to newest)
        tb: TracebackType | None = e.__traceback__
        frames = []
        count = 0
        # Walk to oldest
        stack = []
        while tb is not None:
            stack.append((tb.tb_frame, tb.tb_lineno))
            tb = tb.tb_next
        # Oldest -> newest
        for frame, lineno in stack:
            if count >= max_frames:
                frames.append({"truncated": f"reached max_frames={max_frames}"})
                break
            frames.append(_create_frame_data(frame, lineno))
            count += 1

        return ExceptionBlock(
            type=type(e).__name__,
            module=getattr(type(e), "__module__", None),
            message=_safe_str(e),
            mechanism=Mechanism(type="python", handled=False),
            frames=frames,
        )

    # ---- build top-level envelope -------------------------------------------

    now = time.time()

    # Create thread info
    threads = ThreadInfo(
        current_thread_id=threading.get_ident(),
        current_thread_name=threading.current_thread().name,
    )

    # Create contexts if requested
    contexts = None
    if include_env:
        contexts = Contexts(
            os=OSContext(
                name=platform.system(),
                version=platform.version(),
                release=platform.release(),
                platform=platform.platform(),
            ),
            runtime=RuntimeContext(
                name="CPython",
                version=platform.python_version(),
                build=platform.python_build(),
                implementation=platform.python_implementation(),
            ),
            process=ProcessContext(
                pid=os.getpid(),
                argv=sys.argv,
                executable=sys.executable,
                cwd=os.getcwd(),
            ),
        )

    # Build exception chain
    exceptions = []
    for ex, relation in _iter_chain(exc):
        block = _create_exception_block(ex)
        block.relation = relation
        exceptions.append(block)

    return ExceptionReport(
        event_id=uuid.uuid4().hex,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now))
        + f".{int((now % 1) * 1000):03d}Z",
        level=level,
        platform="python",
        logger=None,
        transaction=None,  # you can set this to current operation/name
        server_name=socket.gethostname(),
        release=None,  # set your app version here
        environment=None,  # e.g., "production", "staging"
        tags={k: str(v)[:128] for (k, v) in (tags or {}).items()},
        extra=_serialize(context or {}, depth=0) if context else {},
        threads=threads,
        exceptions=exceptions,
        contexts=contexts,
    )


def format_exception_report(
    report: ExceptionReport, *, include_env: bool = True, include_vars: bool = True
) -> str:
    """
    Format a structured exception report into human-readable text.

    Parameters:
        report: ExceptionReport instance returned by create_exception_report()
        include_env: Whether to include environment/system information
        include_vars: Whether to include local variables for each frame

    Returns:
        A formatted string suitable for logging or display
    """
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("EXCEPTION REPORT")
    lines.append("=" * 80)
    lines.append(f"Event ID: {report.event_id}")
    lines.append(f"Timestamp: {report.timestamp}")
    lines.append(f"Level: {report.level.upper()}")
    lines.append(f"Server: {report.server_name}")

    # Tags
    if report.tags:
        lines.append(f"Tags: {', '.join(f'{k}={v}' for k, v in report.tags.items())}")

    # Context/Extra info
    if report.extra:
        lines.append("\nContext:")
        for key, value in report.extra.items():
            lines.append(f"  {key}: {_format_value(value)}")

    # Environment info
    if include_env and report.contexts:
        lines.append("\nEnvironment:")

        if report.contexts.os:
            os_info = report.contexts.os
            lines.append(f"  OS: {os_info.name} {os_info.version}")
            lines.append(f"  Platform: {os_info.platform}")

        if report.contexts.runtime:
            rt_info = report.contexts.runtime
            lines.append(f"  Python: {rt_info.implementation} {rt_info.version}")

        if report.contexts.process:
            proc_info = report.contexts.process
            lines.append(f"  PID: {proc_info.pid}")
            lines.append(f"  CWD: {proc_info.cwd}")

    # Thread info
    lines.append(
        f"\nThread: {report.threads.current_thread_name} (ID: {report.threads.current_thread_id})"
    )

    # Exception chain
    if report.exceptions:
        lines.append("\n" + "=" * 80)
        lines.append("EXCEPTION CHAIN")
        lines.append("=" * 80)

        for i, exc_block in enumerate(report.exceptions):
            if i > 0:
                lines.append("\n" + "-" * 60)
                if exc_block.relation == "cause":
                    lines.append(
                        "The above exception was the direct cause of the following exception:"
                    )
                elif exc_block.relation == "context":
                    lines.append(
                        "During handling of the above exception, another exception occurred:"
                    )
                lines.append("-" * 60)

            # Exception info
            if exc_block.module and exc_block.module != "builtins":
                full_type = f"{exc_block.module}.{exc_block.type}"
            else:
                full_type = exc_block.type

            lines.append(f"\n{full_type}: {exc_block.message}")

            # Traceback frames
            if exc_block.frames:
                lines.append("\nTraceback (most recent call last):")

                for frame in exc_block.frames:
                    if isinstance(frame, dict) and "truncated" in frame:
                        lines.append(f"  ... {frame['truncated']}")
                        continue

                    # At this point, frame should be a FrameData instance
                    if not isinstance(frame, FrameData):
                        continue  # Skip if not FrameData for safety

                    # Mark external frames
                    marker = "  " if frame.in_app else "  [external] "

                    # Module info if available
                    location = f"{frame.module}." if frame.module else ""
                    location += frame.function

                    lines.append(
                        f'{marker}File "{frame.filename}", line {frame.lineno}, in {location}'
                    )

                    # Source context
                    if frame.context_line is not None:
                        lines.append(f"{marker}  {frame.context_line.strip()}")

                    # Show surrounding context if available
                    if frame.pre_context or frame.post_context:
                        lines.append(f"{marker}  Source context:")

                        # Pre-context
                        for j, line in enumerate(frame.pre_context):
                            line_num = (
                                (frame.pre_context_lineno + j)
                                if frame.pre_context_lineno
                                else "?"
                            )
                            lines.append(f"{marker}    {line_num:4d} | {line}")

                        # Current line (highlighted)
                        if frame.context_line is not None:
                            lines.append(
                                f"{marker}  > {frame.lineno:4d} | {frame.context_line}"
                            )

                        # Post-context
                        for j, line in enumerate(frame.post_context):
                            line_num = (
                                frame.lineno + j + 1
                                if isinstance(frame.lineno, int)
                                else "?"
                            )
                            lines.append(f"{marker}    {line_num:4d} | {line}")

                    # Local variables
                    if include_vars and frame.vars and isinstance(frame.vars, dict):
                        lines.append(f"{marker}  Local variables:")
                        for var_name, var_value in frame.vars.items():
                            if var_name.startswith("__"):
                                continue  # Skip most dunder vars
                            formatted_value = _format_value(var_value, indent=8)
                            lines.append(f"{marker}    {var_name} = {formatted_value}")

                    if frame.vars_error:
                        lines.append(f"{marker}  Variables error: {frame.vars_error}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def _format_value(value, indent: int = 0) -> str:
    """Format a serialized value for display."""
    indent_str = " " * indent

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


# ----------------------------- Example usage ---------------------------------
if __name__ == "__main__":

    def boom(n):
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
