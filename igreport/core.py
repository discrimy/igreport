"""Core functionality for exception reporting."""

import sys
import os
import uuid
import time
import socket
import platform
import linecache
import threading
from typing import Any, Optional, Dict, List, Union, Generator, Tuple
from types import TracebackType, FrameType
from collections.abc import Mapping, Sequence

from .constants import REDACT_DEFAULT_KEYS
from .models import (
    ExceptionReport,
    ExceptionBlock,
    FrameData,
    Mechanism,
    ThreadInfo,
    OSContext,
    RuntimeContext,
    ProcessContext,
    Contexts,
)
from .utils import _filter_locals_for_report


def create_exception_report(
    exc: BaseException | None = None,
    /,
    *,
    context: Dict[str, Any] | None = None,
    tags: Dict[str, str] | None = None,
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

    def _is_primitive(x: Any) -> bool:
        return isinstance(x, (type(None), bool, int, float, str))

    def _safe_str(x: Any) -> str:
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

    def _serialize(obj: Any, depth: int = 0) -> Any:
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

    def _source_context(
        abs_path: str, lineno: int, ctx_lines: int
    ) -> tuple[List[tuple[int, str]], str, List[tuple[int, str]]]:
        if not abs_path or lineno is None:
            return [], "", []
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

    def _create_frame_data(frame: FrameType, lineno: int) -> FrameData:
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

    def _iter_chain(
        e: BaseException,
    ) -> Generator[Tuple[BaseException, str], None, None]:
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
        frames: List[Union[FrameData, Dict[str, str]]] = []
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