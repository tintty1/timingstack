# TimingStack

A simple yet powerful timing library for Python that helps you measure and analyze code performance with nested timer support.

## Why TimingStack?

I built TimingStack because I needed something more than just basic timing functions. I wanted to:

- Track nested operations and see how they relate to each other
- Get aggregated statistics across multiple runs
- Use timers in different ways (decorators, context managers, manual)
- Handle async/await code gracefully
- Keep memory usage under control with bounded collections

## Features

- **Multiple usage patterns**: Decorator, context manager, or manual start/stop
- **Async/await support**: Works with both sync and async code
- **Nested timers**: Track parent-child relationships between operations
- **Rich statistics**: See counts, totals, averages, min/max, and self-time analysis
- **Memory safe**: Uses bounded lists to prevent memory leaks
- **Configurable**: Adjust time units, precision, error handling, and reporters
- **Global disable**: Turn off all timing for zero production overhead
- **Thread-safe**: Works in multi-threaded environments
- **Pluggable reporters**: Support for custom output formats via TimerReporter protocol
- **Programmatic access**: Get timing data as dictionaries for custom processing

## Installation

```bash
pip install timingstack
```

## Quick Start

### As a decorator (most common)

```python
from timingstack import Timer

@Timer.measure
def process_data():
    # your code here
    time.sleep(0.1)
    return "processed"

result = process_data()
Timer.print_report()
```

### Custom timer names

You can specify custom names for your timers:

```python
# Using Timer.measure with custom name
@Timer.measure("data_processing")
def process_data():
    return "processed"

# Using Timer constructor directly
@Timer("user_authentication")
def authenticate_user():
    return authenticate()

# All three styles work the same
@Timer.measure                # Uses function name: "my_function"
def my_function():
    pass

@Timer.measure()              # Uses function name: "my_function"
def my_function():
    pass

@Timer.measure("custom_name") # Uses custom name: "custom_name"
def my_function():
    pass
```

### As a context manager

```python
with Timer("database_query"):
    # your code here
    results = db.query("SELECT * FROM users")
```

### With async code

```python
@Timer.measure("api_fetch")
async def fetch_data():
    async with Timer("http_request"):
        response = await httpx.get("https://api.example.com")
    return response.json()

result = await fetch_data()
```

### Manual start/stop

```python
Timer.start("complex_operation")
# do some work
Timer.end("complex_operation")
```

## Nested Timers

This is where TimingStack shines:

```python
@Timer.measure("order_processing")
def process_order():
    with Timer("validate_order"):
        validate(order_data)

    with Timer("calculate_price"):
        price = calculate_price(order_data)

    with Timer("save_to_database"):
        with Timer("create_connection"):
            conn = db.connect()

        with Timer("execute_query"):
            db.execute(conn, query)
```

When you run this, you'll see how long each step took AND how much time was spent in child operations.

## Statistics and Reporting

Get timing statistics with:

```python
Timer.print_report()
```

### Custom Reporters

TimingStack supports pluggable reporters for different output formats:

```python
from timingstack import TimerReporter, configure

# Use the default console reporter (included)
# configure(reporter=ConsoleTimerReporter())

# Create custom reporters by implementing the TimerReporter protocol
class MyCustomReporter:
    def generate_report(self, timer_stats, timer_counts, config):
        # Custom reporting logic
        pass

# Set custom reporter
# configure(reporter=MyCustomReporter())
```

## Configuration

Customize how TimingStack behaves:

```python
from timingstack import configure, ErrorHandling

# Show times in seconds with 6 decimal places
configure(time_unit="seconds", precision=6)

# Change error handling
configure(on_mismatch=ErrorHandling.RAISE)  # or WARN, IGNORE

# Adjust memory limits
configure(max_length=1000)  # Keep only 1000 root timers

# Disable timers (useful for production)
configure(enabled=False)    # or True to re-enable

# Set custom reporter
configure(reporter=ConsoleTimerReporter())  # or None to disable reporting
```

## Enable/Disable Timers

Perfect for production environments where you want zero overhead:

```python
from timingstack import configure, Timer

# Disable all timing (no performance overhead)
configure(enabled=False)

@Timer.measure  # This decorator does nothing now
def production_function():
    return expensive_operation()

with Timer("this_is_ignored"):  # This context manager does nothing
    do_something()

# Re-enable for debugging
configure(enabled=True)
```

## Error Handling

TimingStack can handle timer mismatches in three ways:

```python
from timingstack import configure, ErrorHandling

# Warn about mismatches (default)
configure(on_mismatch=ErrorHandling.WARN)

# Raise exceptions for mismatches
configure(on_mismatch=ErrorHandling.RAISE)

# Ignore mismatches silently
configure(on_mismatch=ErrorHandling.IGNORE)
```

## Memory Management

TimingStack uses bounded lists to prevent memory leaks:

```python
# Keep only the last 100 root timers
configure(max_length=100)

# When the limit is reached, old timers are automatically removed
# and a warning is logged
```

## Advanced Usage

### Multiple Timer Stacks

Sometimes you need separate timing contexts:

```python
from timingstack import TimerStack

web_stack = TimerStack()
bg_stack = TimerStack()

# Use them independently
web_stack.start("http_request")
bg_stack.start("file_processing")

# Get separate reports
web_stack.print_report()
bg_stack.print_report()
```

### Timer Counts and Statistics

Access timing data programmatically:

```python
stack = TimerStack()
# ... run some timers ...

# Get call counts for each timer name
counts = stack.get_timer_counts()
# {'http_request': 15, 'database_query': 8, 'validation': 45}

# Get detailed timing statistics
stats = Timer.collect_stats()
# Returns list of timer dictionaries with full hierarchy
```

## Examples

Check out the `example.py` file for comprehensive examples showing all the features.

## Common Patterns

### Performance Profiling

```python
def profile_function():
    configure(time_unit="milliseconds", precision=3)

    @Timer.measure
    def function_to_profile():
        # your function code
        pass

    # Run it multiple times to get good stats
    for _ in range(100):
        function_to_profile()

    Timer.print_report()
```

### Database Operations

```python
def save_user(user_data):
    with Timer("save_user"):
        with Timer("validate"):
            validate_user_data(user_data)

        with Timer("hash_password"):
            hashed = hash_password(user_data.password)

        with Timer("db_insert"):
            db.execute("INSERT INTO users ...", user_data)
```

### API Endpoints

```python
@Timer.measure("user_endpoint")
@app.route("/api/users/<int:user_id>")
def get_user(user_id):
    with Timer("database_lookup"):
        user = db.get_user(user_id)

    with Timer("serialize_response"):
        return jsonify(user.to_dict())
```

## That's It!

TimingStack is designed to be simple to use but powerful enough for real-world performance analysis. Start with decorators, then explore nested timers, custom reporters, and programmatic access as you need them.

## API Reference

### Core Classes

- **`Timer`**: Main timer class supporting decorators, context managers, and manual timing
- **`TimerStack`**: Container for managing timer hierarchies and statistics
- **`TimerConfig`**: Configuration object for global timer settings
- **`TimerContext`**: Represents a single timer execution with start/end times

### Reporting Classes

- **`TimerReporter`**: Protocol for implementing custom reporters
- **`ConsoleTimerReporter`**: Default console-based reporter implementation
- **`TimeUnit`**: Enum for supported time units (seconds, milliseconds, microseconds)

### Configuration

- **`configure(**kwargs)`\*\*: Update global timer configuration
- **`get_config()`**: Get current global configuration

### Enums

- **`ErrorHandling`**: Options for handling timer mismatches (WARN, RAISE, IGNORE)

If you run into issues or have ideas for improvements, feel free to open an issue or submit a pull request.
