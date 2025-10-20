import asyncio
import logging
import threading
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, Any, Optional, TypeVar, cast, overload

if TYPE_CHECKING:
    from .reporting import TimerReporter

logger = logging.getLogger("timingstack")

F = TypeVar("F", bound=Callable[..., Any])


class ErrorHandling(Enum):
    WARN = "warn"
    IGNORE = "ignore"
    RAISE = "raise"


@dataclass
class TimerConfig:
    """Global settings for all timers"""

    enabled: bool = True
    on_mismatch: ErrorHandling = ErrorHandling.WARN
    warn_unclosed: bool = True
    auto_close_unclosed: bool = True
    time_unit: str = "milliseconds"
    precision: int = 2
    logger_name: str = "timingstack"
    log_level: str = "WARNING"
    max_length: int = 1000
    reporter: Optional["TimerReporter"] = None


_config: TimerConfig | None = None
_config_lock = threading.RLock()


def get_config() -> TimerConfig:
    """
    Get the global timer configuration.

    Returns:
        TimerConfig: The current global configuration
    """
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                from .reporting import ConsoleTimerReporter

                _config = TimerConfig(reporter=ConsoleTimerReporter())
    return _config


def configure(**kwargs) -> None:
    """
    Configure the global timer settings.

    Args:
        **kwargs: Configuration options to update. Valid keys are:
            - enabled: bool (whether timers are active)
            - on_mismatch: ErrorHandling enum value
            - warn_unclosed: bool
            - auto_close_unclosed: bool
            - time_unit: str ("seconds", "milliseconds", "microseconds")
            - precision: int
            - logger_name: str
            - log_level: str
            - max_length: int (maximum number of items in bounded lists)
            - reporter: TimerReporter implementation (set to None to disable reporting)
    """
    global _config
    with _config_lock:
        current_config = get_config()

        reporter = current_config.reporter
        if "reporter" in kwargs and kwargs["reporter"] is None:
            reporter = None

        new_config = TimerConfig(
            enabled=kwargs.get("enabled", current_config.enabled),
            on_mismatch=kwargs.get("on_mismatch", current_config.on_mismatch),
            warn_unclosed=kwargs.get("warn_unclosed", current_config.warn_unclosed),
            auto_close_unclosed=kwargs.get(
                "auto_close_unclosed", current_config.auto_close_unclosed
            ),
            time_unit=kwargs.get("time_unit", current_config.time_unit),
            precision=kwargs.get("precision", current_config.precision),
            logger_name=kwargs.get("logger_name", current_config.logger_name),
            log_level=kwargs.get("log_level", current_config.log_level),
            max_length=kwargs.get("max_length", current_config.max_length),
            reporter=reporter,
        )
        _config = new_config


@dataclass
class TimerContext:
    name: str
    start_time: float
    end_time: float | None = None
    parent: Optional["TimerContext"] = None
    children: list["TimerContext"] = field(default_factory=list)

    @property
    def duration(self) -> float | None:
        """Total duration including children's durations"""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def self_duration(self) -> float | None:
        """Duration of self, excluding children"""
        if self.duration is None:
            return None
        children_duration = sum(c.duration or 0 for c in self.children)
        return self.duration - children_duration

    def is_complete(self) -> bool:
        """Check if timer is complete"""
        return self.end_time is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict"""
        return {
            "name": self.name,
            "duration": self.duration,
            "self_duration": self.self_duration,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "children": [c.to_dict() for c in self.children],
        }


class TimerStack:
    """Manage the stack of active timers"""

    def __init__(self):
        max_length = get_config().max_length
        # Use bounded list for root timers to prevent memory leaks
        self.root_timers: BoundedList = BoundedList(max_length=max_length)
        # Use regular list for active stack to preserve timer hierarchy
        self.active_stack: list[TimerContext] = []
        # Track timer call counts by name
        self.timer_counts: dict[str, int] = {}

    def start(self, name: str) -> TimerContext:
        """Start a new timer."""
        ctx = TimerContext(name=name, start_time=time.perf_counter())

        # Track timer call count
        self.timer_counts[name] = self.timer_counts.get(name, 0) + 1

        if len(self.active_stack) > 0:
            # not root timer
            parent = self.active_stack[-1]
            ctx.parent = parent
            parent.children.append(ctx)
        else:
            # No parent -> root timer
            self.root_timers.append(ctx)

        self.active_stack.append(ctx)
        return ctx

    def end(self, name: str | None = None) -> TimerContext | None:
        """
        End a timer

        Args:
            name: Timer name to end. If None, end the most recent timer.

        Returns:
            The ended TimerContext, or None if no match found.
        """
        if len(self.active_stack) == 0:
            self._handle_error(f"Timer.end('{name}') called but no active timers.")
            return None

        if name is None:
            # end the most recent timer
            ctx = self.active_stack.pop()
            ctx.end_time = time.perf_counter()
            return ctx

        # find matching timer
        for i in range(len(self.active_stack) - 1, -1, -1):
            if self.active_stack[i].name == name:
                ctx = self.active_stack.pop(i)
                ctx.end_time = time.perf_counter()

                # close any nested orphaned timers
                if i < len(self.active_stack):
                    # Get orphaned timers and close them
                    for j in range(i, len(self.active_stack)):
                        orphan = self.active_stack[j]
                        if not orphan.is_complete():
                            orphan.end_time = time.perf_counter()
                            logger.warning(
                                f"Timer '{orphan.name}' was automatically closed "
                                f"because parent '{name}' has ended."
                            )
                    # Remove orphaned timers
                    del self.active_stack[i:]

                return ctx

        # no matching timer found
        self._handle_error(
            f"Timer.end('{name}') called but no matching start found. "
            f"Active timers: {[t.name for t in self.active_stack]}"
        )
        return None

    def close_all(self) -> None:
        """Close all active timers"""
        current_time = time.perf_counter()
        # Iterate backwards without creating new list
        for i in range(len(self.active_stack) - 1, -1, -1):
            ctx = self.active_stack[i]
            if not ctx.is_complete():
                ctx.end_time = current_time
                if get_config().warn_unclosed:
                    logger.warning(f"Timer '{ctx.name}' was never closed. Auto close now.")
        self.active_stack.clear()

    def reset(self) -> None:
        self.root_timers.clear()
        self.active_stack.clear()
        self.timer_counts.clear()

    def _handle_error(self, message: str) -> None:
        if get_config().on_mismatch == ErrorHandling.WARN:
            logger.warning(message)
        elif get_config().on_mismatch == ErrorHandling.RAISE:
            raise ValueError(message)
        # else: pass

    def get_stats(self) -> list[dict[str, Any]]:
        """Export timing data as a dict"""
        return [root.to_dict() for root in self.root_timers.items()]

    def get_timer_counts(self) -> dict[str, int]:
        """Get call counts for each timer name"""
        return dict(self.timer_counts)

    def print_report(self) -> None:
        """Generate timing report using the configured reporter"""
        config = get_config()
        timer_stats = self._collect_timer_stats()

        # Check if reporter is None and warn user
        if config.reporter is None:
            logger.warning(
                "Timer reporter is set to None. No report will be generated. "
                "Set a reporter using configure(reporter=YourReporterClass())."
            )
            return

        # Use injected reporter for output
        config.reporter.generate_report(timer_stats, self.get_timer_counts(), config)

    def _collect_timer_stats(self) -> dict[str, dict[str, list[float]]]:
        """Collect and aggregate timer statistics"""

        def _collect_all_timers(timers: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Recursively collect all timers in a flat list"""
            all_timers = []
            for timer in timers:
                all_timers.append(timer)
                for child in timer.get("children", []):
                    all_timers.extend(_collect_all_timers([child]))
            return all_timers

        # Collect all timers and group by name
        all_timers = _collect_all_timers(self.get_stats())
        timer_stats = {}  # name -> list of durations, self_durations

        for timer in all_timers:
            name = timer["name"]
            duration = timer.get("duration")
            self_duration = timer.get("self_duration")

            if name not in timer_stats:
                timer_stats[name] = {"durations": [], "self_durations": []}

            if duration is not None:
                timer_stats[name]["durations"].append(duration)
            if self_duration is not None:
                timer_stats[name]["self_durations"].append(self_duration)

        return timer_stats


class BoundedList:
    """A wrapper for list to use in TimerStack"""

    def __init__(self, max_length: int | None = None):
        self._data: list[TimerContext] = []
        self._max_length = max_length or get_config().max_length

    def append(self, item: TimerContext) -> None:
        self._data.append(item)

        if len(self._data) > self._max_length:
            # Remove oldest items from the beginning
            excess = len(self._data) - self._max_length
            logger.warning(
                f"BoundedList exceeded max length of {self._max_length}. "
                f"Removing {excess} oldest item(s)."
            )
            del self._data[:excess]

    def pop(self) -> TimerContext:
        return self._data.pop()

    def pop_at_index(self, index: int) -> TimerContext:
        return self._data.pop(index)

    def truncate(self, new_size: int) -> None:
        if new_size < len(self._data):
            del self._data[new_size:]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> TimerContext:
        return self._data[index]

    def __iter__(self):
        return iter(self._data)

    def clear(self) -> None:
        self._data.clear()

    def items(self) -> list[TimerContext]:
        return self._data


_timer_stack: ContextVar[TimerStack | None] = ContextVar("timer_stack", default=None)


def _get_stack() -> TimerStack:
    """Get or create timer stack for current context"""
    stack = _timer_stack.get()
    if stack is None:
        stack = TimerStack()
        _timer_stack.set(stack)
    return stack


class Timer:
    """
    Timer API, 3 main types

    Usage:
        # As decorator (sync)
        @Timer.measure
        def my_function():
            pass

        # As decorator (async)
        @Timer.measure
        async def my_async_function():
            pass

        # As context manager (sync)
        with Timer("section"):
            code()

        # As async context manager
        async with Timer("section"):
            await async_code()

        # Manual start/stop
        Timer.start("task")
        code()
        Timer.end("task")
    """

    def __init__(self, name: str | None = None):
        """
        Init timer

        Args:
            name: Timer name. If None and used as decorator, uses function name.
        """
        self.name = name
        self._ctx: TimerContext | None = None

    def __enter__(self) -> "Timer":
        if self.name is None:
            raise RuntimeError("`name` must be set")
        self._ctx = Timer.start(self.name)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        Timer.end(self.name)
        return False

    async def __aenter__(self) -> "Timer":
        if self.name is None:
            raise RuntimeError("`name` must be set")
        self._ctx = Timer.start(self.name)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        Timer.end(self.name)
        return False

    def __call__(self, func: F) -> F:
        """Decorator usage, supports both sync and async functions."""
        name = self.name or func.__name__

        # check if function is a coroutine function (async)
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with Timer(name):
                    return await func(*args, **kwargs)

            return cast(F, async_wrapper)
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with Timer(name):
                    return func(*args, **kwargs)

            return cast(F, sync_wrapper)

    @staticmethod
    def start(name: str) -> TimerContext | None:
        """
        Start a timer manually.

        Args:
            name: Timer name

        Returns:
            TimerContext for the started timer, or None if disabled
        """
        if not get_config().enabled:
            return None

        stack = _get_stack()
        return stack.start(name)

    @staticmethod
    def end(name: str | None = None) -> TimerContext | None:
        """
        End a timer manually

        Args:
            name: Timer name to end. If None, ends the most recent timer.

        Returns:
            The ended TimerContext, or None if no match found
        """
        if not get_config().enabled:
            return None

        stack = _get_stack()
        return stack.end(name)

    @staticmethod
    def reset() -> None:
        """Reset all timing data in current context."""
        if not get_config().enabled:
            return
        stack = _get_stack()
        stack.reset()

    @staticmethod
    def collect_stats() -> None | list[dict[str, Any]]:
        """Collect stats."""
        if not get_config().enabled:
            return
        stack = _get_stack()
        return stack.get_stats()

    @staticmethod
    def print_report() -> None:
        """Print formatted report for current context."""
        if not get_config().enabled:
            print("TimingStack is currently disabled. Enable with configure(enabled=True).")
            return

        stack = _get_stack()
        stack.print_report()

    @staticmethod
    @overload
    def measure(name_or_func: None = None) -> Callable[[F], F]:
        """
        Decorator to measure function execution time - no args version.

        Usage:
            @Timer.measure()
            def my_function():
                pass
        """
        ...

    @staticmethod
    @overload
    def measure(name_or_func: str) -> Callable[[F], F]:
        """
        Decorator to measure function execution time - with custom name.

        Usage:
            @Timer.measure("custom_name")
            def my_function():
                pass
        """
        ...

    @staticmethod
    @overload
    def measure(name_or_func: F) -> F:
        """
        Decorator to measure function execution time - direct function version.

        Usage:
            @Timer.measure  # no parentheses
            def my_function():
                pass
        """
        ...

    @staticmethod
    def measure(name_or_func: str | F | None = None) -> F | Callable[[F], F]:
        """
        Decorator to measure function execution time

        Usage:
            @Timer.measure()
            def my_function():
                pass

            @Timer.measure("custom_name")
            def my_function():
                pass

            @Timer.measure  # no parentheses
            def my_function():
                pass

        Args:
            name_or_func: Custom timer name, function to measure, or None.
                         If None and used as @Timer.measure(), uses function name.
                         If string, uses as custom name with @Timer.measure("name").

        Returns:
            Wrapped function or decorator
        """
        if isinstance(name_or_func, str):
            # Called with custom name: @Timer.measure("custom_name")
            custom_name = name_or_func

            def decorator(func: F) -> F:
                return Timer(custom_name)(func)

            return decorator
        elif callable(name_or_func):
            # Called directly with function: @Timer.measure
            func = cast(F, name_or_func)
            return Timer(func.__name__)(func)
        else:
            # Called with no args: @Timer.measure()
            def decorator(func: F) -> F:
                return Timer(func.__name__)(func)

            return decorator
