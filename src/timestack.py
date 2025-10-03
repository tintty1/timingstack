import time
import asyncio
import logging
from contextvars import ContextVar
from typing import Optional, List, Any, Dict, Callable, TypeVar, cast
from functools import wraps
from dataclasses import dataclass, field
from enum import Enum


logger = logging.getLogger("timestack")

F = TypeVar("F", bound=Callable[..., Any])


class ErrorHandling(Enum):
    WARN = "warn"
    IGNORE = "ignore"
    RAISE = "raise"


@dataclass
class TimerConfig:
    """Global settings for all timers"""

    on_mismatch: ErrorHandling = ErrorHandling.WARN
    warn_unclosed: bool = True
    auto_close_unclosed: bool = True
    time_unit: str = "milliseconds"
    precision: int = 2
    logger_name: str = "timestack"
    log_level: str = "WARNING"


_config = TimerConfig()

# TODO: add util function to configure the timer??


@dataclass
class TimerContext:
    name: str
    start_time: float
    end_time: Optional[float] = None
    parent: Optional["TimerContext"] = None
    children: List["TimerContext"] = field(default_factory=list)

    @property
    def duration(self) -> Optional[float]:
        """Total duration including children's durations"""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def self_duration(self) -> Optional[float]:
        """Duration of self, excluding children"""
        if self.duration is None:
            return None
        children_duration = sum(c.duration or 0 for c in self.children)
        return self.duration - children_duration

    def is_complete(self) -> bool:
        """Check if timer is complete"""
        return self.end_time is not None

    def to_dict(self) -> Dict[str, Any]:
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
        # TODO: should use some kind of bounded list here, deque, maybe?
        # a list to track the root timers
        self.root_timers: List[TimerContext] = []
        # list of childs, nested inside root timer
        self.active_stack: List[TimerContext] = []

    def start(self, name: str) -> TimerContext:
        """Start a new timer."""
        ctx = TimerContext(name=name, start_time=time.perf_counter())

        if self.active_stack:
            # not root timer
            parent = self.active_stack[-1]
            ctx.parent = parent
            parent.children.append(ctx)
        else:
            # No parent -> root timer
            self.root_timers.append(ctx)

        self.active_stack.append(ctx)
        return ctx

    def end(self, name: Optional[str] = None) -> Optional[TimerContext]:
        """
        End a timer

        Args:
            name: Timer name to end. If None, end the most recent timer.

        Returns:
            The ended TimerContext, or None if no match found.
        """
        if not self.active_stack:
            self._handle_error(f"Timer.end('{name}') called but no active timers.")
            return None

        if name is None:
            # end the most recent timer
            ctx = self.active_stack.pop()
            ctx.end_time = time.perf_counter()
            return ctx

        # find matching timer
        # TODO: check this
        for i in range(len(self.active_stack) - 1, -1, -1):
            if self.active_stack[i].name == name:
                ctx = self.active_stack.pop(i)
                ctx.end_time = time.perf_counter()

                # close any nested orphaned timers
                if i < len(self.active_stack):
                    # import pdb; pdb.set_trace()
                    orphaned = self.active_stack[i:]
                    self.active_stack = self.active_stack[:i]
                    for orphan in orphaned:
                        if not orphan.is_complete():
                            orphan.end_time = time.perf_counter()
                            logger.warning(
                                f"Timer '{orphan.name}' was automatically closed bcs parent '{name}' has ended."
                            )

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
        for ctx in reversed(self.active_stack):
            if not ctx.is_complete():
                ctx.end_time = current_time
                if _config.warn_unclosed:
                    logger.warning(
                        f"Timer '{ctx.name}' was never closed. Auto close now."
                    )
        self.active_stack.clear()

    def reset(self) -> None:
        self.root_timers.clear()
        self.active_stack.clear()

    def _handle_error(self, message: str) -> None:
        if _config.on_mismatch == ErrorHandling.WARN:
            logger.warning(message)
        elif _config.on_mismatch == ErrorHandling.RAISE:
            raise ValueError(message)
        # else: pass

    def get_stats(self) -> List[Dict[str, Any]]:
        """Export timing data as a dict"""
        return [root.to_dict() for root in self.root_timers]

    def print_report(self) -> None:
        """Print formatted timing report"""
        print("\n" + "=" * 50)
        print("TIMING REPORT")
        print("=" * 50)

        def _convert_duration(duration: float) -> float:
            if _config.time_unit == "seconds":
                return duration
            elif _config.time_unit == "milliseconds":
                return duration * 1000
            elif _config.time_unit == "microseconds":
                return duration * 1000000
            else:
                return duration * 1000  # default to milliseconds

        def _get_unit_suffix() -> str:
            if _config.time_unit == "seconds":
                return "s"
            elif _config.time_unit == "milliseconds":
                return "ms"
            elif _config.time_unit == "microseconds":
                return "μs"
            else:
                return "ms"

        def _print_timer(timer: Dict[str, Any], indent: int = 0) -> None:
            prefix = "  " * indent
            duration = _convert_duration(timer.get("duration", 0) or 0)
            self_duration = _convert_duration(timer.get("self_duration", 0) or 0)
            unit = _get_unit_suffix()

            print(
                f"{prefix}- {timer['name']}: {duration:.{_config.precision}f}{unit} "
                f"(self: {self_duration:.{_config.precision}f}{unit})"
            )

            for child in timer.get("children", []):
                _print_timer(child, indent + 1)

        for root in self.get_stats():
            _print_timer(root)

        print("=" * 50)


_timer_stack: ContextVar[Optional[TimerStack]] = ContextVar("timer_stack", default=None)


# TODO: verify if this is thread safe


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

    def __init__(self, name: Optional[str] = None):
        """
        Init timer

        Args:
            name: Timer name. If None and used as decorator, uses function name.
        """
        self.name = name
        self._ctx: Optional[TimerContext] = None

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
    def start(name: str) -> TimerContext:
        """
        Start a timer manually.

        Args:
            name: Timer name

        Returns:
            TimerContext for the started timer
        """
        stack = _get_stack()
        return stack.start(name)

    @staticmethod
    def end(name: Optional[str] = None) -> Optional[TimerContext]:
        """
        End a timer manually

        Args:
            name: Timer name to end. If None, ends the most recent timer.

        Returns:
            The ended TimerContext, or None if no match found
        """
        stack = _get_stack()
        return stack.end(name)

    @staticmethod
    def reset() -> None:
        """Reset all timing data in current context."""
        stack = _get_stack()
        stack.reset()

    @staticmethod
    def print_report() -> None:
        """Print formatted report for current context."""
        stack = _get_stack()
        stack.print_report()

    @staticmethod
    def measure(func: F) -> F:
        """
        Decorator to measure function execution time

        Args:
            func: Function to measure

        Returns:
            Wrapped function
        """
        return Timer(func.__name__)(func)
