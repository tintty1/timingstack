"""Reporting abstractions and implementations for TimingStack."""

from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .timingstack import TimerConfig


class TimeUnit(Enum):
    """Supported time units for reporting"""

    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    MICROSECONDS = "microseconds"


class TimeConverter(Protocol):
    """Protocol for time unit conversion"""

    def convert(self, duration: float, unit: TimeUnit) -> float: ...
    def get_unit_suffix(self, unit: TimeUnit) -> str: ...


class TimerReporter(Protocol):
    """Protocol for timer reporting output"""

    def generate_report(
        self,
        timer_stats: dict[str, dict[str, list[float]]],
        timer_counts: dict[str, int],
        config: "TimerConfig",
    ) -> None: ...


class DefaultTimeConverter:
    """Default implementation of time unit conversion"""

    def convert(self, duration: float, unit: TimeUnit) -> float:
        """Convert duration to specified time unit"""
        if unit == TimeUnit.SECONDS:
            return duration
        elif unit == TimeUnit.MILLISECONDS:
            return duration * 1000
        elif unit == TimeUnit.MICROSECONDS:
            return duration * 1000000
        else:
            return duration * 1000  # default to milliseconds

    def get_unit_suffix(self, unit: TimeUnit) -> str:
        """Get the suffix string for the time unit"""
        if unit == TimeUnit.SECONDS:
            return "s"
        elif unit == TimeUnit.MILLISECONDS:
            return "ms"
        elif unit == TimeUnit.MICROSECONDS:
            return "μs"
        else:
            return "ms"


class ConsoleTimerReporter:
    """Default console-based timer reporter"""

    def __init__(self, time_converter: TimeConverter | None = None):
        self.time_converter = time_converter or DefaultTimeConverter()

    def generate_report(
        self,
        timer_stats: dict[str, dict[str, list[float]]],
        timer_counts: dict[str, int],
        config: "TimerConfig",
    ) -> None:
        """Generate a console timing report"""
        print("\n" + "=" * 50)
        print("TIMING REPORT")
        print("=" * 50)

        if not timer_stats:
            print("No timers recorded.")
            print("=" * 50)
            return

        # Convert string time_unit to TimeUnit enum
        time_unit = TimeUnit(config.time_unit)
        unit_suffix = self.time_converter.get_unit_suffix(time_unit)
        precision = config.precision

        # Print aggregated statistics for each timer
        print(f"{'Timer':<20} {'Count':<8} {'Total':<12} {'Avg':<12} {'Min':<12} {'Max':<12}")
        print("-" * 84)

        for name, stats in sorted(timer_stats.items()):
            durations = stats["durations"]

            if durations:
                total = sum(durations)
                avg = total / len(durations)
                min_time = min(durations)
                max_time = max(durations)

                total_str = (
                    f"{self.time_converter.convert(total, time_unit):.{precision}f}{unit_suffix}"
                )
                avg_str = (
                    f"{self.time_converter.convert(avg, time_unit):.{precision}f}{unit_suffix}"
                )
                min_str = (
                    f"{self.time_converter.convert(min_time, time_unit):.{precision}f}{unit_suffix}"
                )
                max_str = (
                    f"{self.time_converter.convert(max_time, time_unit):.{precision}f}{unit_suffix}"
                )

                print(
                    f"{name:<20} {len(durations):<8} "
                    f"{total_str:>12} {avg_str:>12} {min_str:>12} {max_str:>12}"
                )

        # Print self-time statistics if we have data
        has_self_time = any(stats["self_durations"] for stats in timer_stats.values())
        if has_self_time:
            print(
                f"\n{'Timer':<20} {'Count':<8} {'Self Total':<12} "
                f"{'Self Avg':<12} {'Self Min':<12} {'Self Max':<12}"
            )
            print("-" * 84)

            for name, stats in sorted(timer_stats.items()):
                self_durations = stats["self_durations"]

                if self_durations:
                    total_self = sum(self_durations)
                    avg_self = total_self / len(self_durations)
                    min_self = min(self_durations)
                    max_self = max(self_durations)

                    total_str = (
                        f"{self.time_converter.convert(total_self, time_unit):.{precision}f}"
                        f"{unit_suffix}"
                    )
                    avg_str = (
                        f"{self.time_converter.convert(avg_self, time_unit):.{precision}f}"
                        f"{unit_suffix}"
                    )
                    min_str = (
                        f"{self.time_converter.convert(min_self, time_unit):.{precision}f}"
                        f"{unit_suffix}"
                    )
                    max_str = (
                        f"{self.time_converter.convert(max_self, time_unit):.{precision}f}"
                        f"{unit_suffix}"
                    )

                    print(
                        f"{name:<20} {len(self_durations):<8} "
                        f"{total_str:>12} {avg_str:>12} {min_str:>12} {max_str:>12}"
                    )

        print("=" * 50)
