"""Data models for exception reporting."""

from dataclasses import dataclass, asdict
from typing import Any, Optional, Dict, List, Union


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
    build: tuple[str, str]
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