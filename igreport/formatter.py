"""Formatting utilities for exception reports."""

from typing import Any
from .models import ExceptionReport, FrameData
from .utils import _format_value


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