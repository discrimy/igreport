# igreport

A library for structured exception reporting and formatting, providing detailed, JSON-serializable exception reports similar to what Sentry provides, along with human-readable formatting capabilities.

## Features

- **Structured Exception Reports**: Create detailed, JSON-serializable exception reports
- **Human-Readable Formatting**: Convert reports to well-formatted text for logging
- **Exception Chain Support**: Handle chained exceptions (both `__cause__` and `__context__`)
- **Local Variable Capture**: Safely capture and serialize local variables with redaction
- **Source Code Context**: Include surrounding source code lines in reports
- **Environment Information**: Capture system, runtime, and process context
- **Memory Efficient**: Configurable limits for frames, string length, and container sizes
- **Security**: Built-in redaction of sensitive data (passwords, keys, tokens)

## Installation

```bash
pip install igreport
```

## Quick Start

```python
import igreport

try:
    # Your code that might raise an exception
    result = 10 / 0
except Exception as e:
    # Create a structured report
    report = igreport.create_exception_report(e)
    
    # Format as human-readable text
    formatted = igreport.format_exception_report(report)
    print(formatted)
    
    # Or get as JSON-serializable dict
    report_dict = report.to_dict()
```

## Advanced Usage

### Custom Context and Tags

```python
import igreport

try:
    # Some operation
    process_user_data(user_id=12345)
except Exception as e:
    report = igreport.create_exception_report(
        e,
        context={
            "user_id": 12345,
            "operation": "process_user_data",
            "request_id": "req-abc123"
        },
        tags={
            "service": "user-service",
            "environment": "production"
        },
        level="error"
    )
```

### Configuration Options

```python
import igreport

try:
    # Your code
    raise ValueError("Something went wrong")
except Exception as e:
    report = igreport.create_exception_report(
        e,
        max_frames=50,           # Limit number of stack frames
        context_lines=5,         # Source code lines around each frame
        capture_locals=True,     # Include local variables
        max_str_len=512,        # Truncate long strings
        max_container_items=25,  # Limit items in lists/dicts
        max_depth=4,            # Limit nested serialization depth
        include_env=True,       # Include system information
        redact_keys={"custom_secret"}  # Additional keys to redact
    )
```

### Custom Redaction

```python
import igreport

# Built-in redacted keys include: password, secret, token, key, etc.
# Add your own sensitive keys:
custom_redactions = {"session_id", "auth_header", "private_data"}

try:
    sensitive_operation()
except Exception as e:
    report = igreport.create_exception_report(
        e,
        redact_keys=custom_redactions,
        capture_locals=True
    )
```

### Formatting Options

```python
import igreport

# Create report
report = igreport.create_exception_report(exception)

# Format with different options
formatted = igreport.format_exception_report(
    report,
    include_env=True,    # Include environment information
    include_vars=True    # Include local variables
)
```

## Library Structure

The library is organized into several modules:

- **`igreport.core`**: Main exception reporting functionality
- **`igreport.models`**: Data classes for structured reports
- **`igreport.formatter`**: Human-readable text formatting
- **`igreport.utils`**: Utility functions for variable filtering and formatting
- **`igreport.constants`**: Configuration constants

## API Reference

### Main Functions

#### `create_exception_report(exc=None, *, context=None, tags=None, level="error", **options)`

Create a structured exception report.

**Parameters:**
- `exc`: Exception instance (if None, uses current exception)
- `context`: Additional context data (dict)
- `tags`: Lightweight labels for filtering (dict)
- `level`: Log level ("error", "warning", etc.)
- `max_frames`: Maximum stack frames to include (default: 200)
- `context_lines`: Source lines around each frame (default: 3)
- `capture_locals`: Include local variables (default: True)
- `redact_keys`: Additional keys to redact (set)
- `max_str_len`: Maximum string length (default: 1024)
- `max_container_items`: Maximum items in containers (default: 50)
- `max_depth`: Maximum serialization depth (default: 3)
- `include_env`: Include environment info (default: True)

**Returns:** `ExceptionReport` instance

#### `format_exception_report(report, *, include_env=True, include_vars=True)`

Format an exception report as human-readable text.

**Parameters:**
- `report`: ExceptionReport instance
- `include_env`: Include environment information
- `include_vars`: Include local variables

**Returns:** Formatted string

### Data Classes

#### `ExceptionReport`
Main report structure containing:
- `event_id`: Unique identifier
- `timestamp`: ISO timestamp
- `level`: Log level
- `exceptions`: List of exception blocks
- `contexts`: Environment contexts
- `threads`: Thread information
- `tags`: User-defined tags
- `extra`: Additional context

#### `ExceptionBlock`
Individual exception information:
- `type`: Exception class name
- `message`: Exception message
- `frames`: Stack frames
- `relation`: Relationship to other exceptions ("exception", "cause", "context")

#### `FrameData`
Stack frame information:
- `filename`: Source file name
- `function`: Function name
- `lineno`: Line number
- `context_line`: Current source line
- `pre_context`/`post_context`: Surrounding source lines
- `vars`: Local variables (if captured)
- `in_app`: Whether frame is from application code

## Security

The library automatically redacts sensitive information from local variables:

### Default Redacted Keys
- `password`, `passwd`, `pwd`
- `secret`, `api_secret`, `client_secret`
- `token`, `access_token`, `refresh_token`, `id_token`
- `key`, `api_key`, `private_key`
- `authorization`, `auth`
- `cookie`, `set-cookie`

### Custom Redaction
You can specify additional keys to redact:

```python
report = igreport.create_exception_report(
    exception,
    redact_keys={"session_id", "user_token", "private_data"}
)
```

## Performance Considerations

- Use `capture_locals=False` for better performance in production
- Adjust `max_frames`, `max_str_len`, and `max_container_items` for memory efficiency
- Set `include_env=False` if environment information is not needed

## Examples

### Basic Error Reporting

```python
import igreport
import json

def divide_numbers(a, b):
    return a / b

try:
    result = divide_numbers(10, 0)
except Exception as e:
    report = igreport.create_exception_report(e)
    
    # Save as JSON
    with open('error_report.json', 'w') as f:
        json.dump(report.to_dict(), f, indent=2)
    
    # Log formatted version
    print(igreport.format_exception_report(report))
```

### Web Application Integration

```python
import igreport
from flask import Flask, request

app = Flask(__name__)

@app.errorhandler(Exception)
def handle_exception(e):
    report = igreport.create_exception_report(
        e,
        context={
            "url": request.url,
            "method": request.method,
            "headers": dict(request.headers),
            "user_agent": request.user_agent.string
        },
        tags={
            "service": "web-api",
            "endpoint": request.endpoint
        }
    )
    
    # Log for debugging
    app.logger.error(igreport.format_exception_report(report))
    
    # Send to monitoring service
    # monitoring_service.send(report.to_dict())
    
    return "Internal Server Error", 500
```

### Chained Exceptions

```python
import igreport

def process_data():
    try:
        # Some operation that fails
        raise ValueError("Data validation failed")
    except ValueError as e:
        # Wrap with more context
        raise RuntimeError("Processing failed") from e

try:
    process_data()
except Exception as e:
    report = igreport.create_exception_report(e)
    formatted = igreport.format_exception_report(report)
    
    # Will show both exceptions with "direct cause" relationship
    print(formatted)
```

## Testing

Run the test suite:

```bash
pytest test_igreport.py -v
```

## License

[License information here]
