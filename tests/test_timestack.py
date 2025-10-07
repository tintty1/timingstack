import asyncio
import time

import pytest

from timingstack import (
    BoundedList,
    ErrorHandling,
    Timer,
    TimerConfig,
    TimerContext,
    TimerStack,
    configure,
    get_config,
)


class TestBoundedList:
    def test_basic_operations(self):
        lst = BoundedList()
        lst.append("item1")
        lst.append("item2")

        assert len(lst) == 2
        assert lst[0] == "item1"
        assert lst[1] == "item2"

        item = lst.pop()
        assert item == "item2"
        assert len(lst) == 1

        lst.clear()
        assert len(lst) == 0

    def test_max_length_enforcement(self):
        # Reset global state first
        import timingstack as ts

        ts._config = None

        # Test with small max length
        configure(max_length=3)
        lst = BoundedList()

        # Add items up to max length
        lst.append("item1")
        lst.append("item2")
        lst.append("item3")

        assert len(lst) == 3
        assert lst[0] == "item1"
        assert lst[1] == "item2"
        assert lst[2] == "item3"

        # Add one more item - should remove oldest
        lst.append("item4")

        # Should still have max_length items
        assert len(lst) == 3
        # Oldest item should be removed
        assert lst[0] == "item2"
        assert lst[1] == "item3"
        assert lst[2] == "item4"

    def test_custom_max_length(self):
        # Reset global state first
        import timingstack as ts

        ts._config = None

        # Configure with higher global max length
        configure(max_length=100)

        # Create BoundedList with custom max length
        lst = BoundedList(max_length=5)

        # Add items beyond custom max length
        for i in range(10):
            lst.append(f"item{i}")

        # Should have exactly custom max_length items
        assert len(lst) == 5
        assert lst[0] == "item5"
        assert lst[4] == "item9"

    def test_multiple_excess_items(self):
        # Reset global state first
        import timingstack as ts

        ts._config = None

        # Test with very small max length
        configure(max_length=2)
        lst = BoundedList()

        # Add many items at once
        lst.append("item1")
        lst.append("item2")
        lst.append("item3")
        lst.append("item4")
        lst.append("item5")

        # Should only have max_length items, with the most recent ones
        assert len(lst) == 2
        assert lst[0] == "item4"
        assert lst[1] == "item5"

    def test_max_length_config_update(self):
        # Reset global state first
        import timingstack as ts

        ts._config = None

        # Start with initial max length
        configure(max_length=3)
        lst = BoundedList()

        lst.append("item1")
        lst.append("item2")
        lst.append("item3")
        assert len(lst) == 3

        # Update max length to smaller value
        configure(max_length=2)
        lst_new = BoundedList()

        lst_new.append("a")
        lst_new.append("b")
        lst_new.append("c")

        assert len(lst_new) == 2
        assert lst_new[0] == "b"
        assert lst_new[1] == "c"

        # Old list should still use its original max length
        assert len(lst) == 3

    def test_boundedlist_with_root_timers(self):
        # Reset global state first
        import timingstack as ts

        ts._config = None

        # Test with small max length to verify root timer limiting
        configure(max_length=2)
        stack = TimerStack()

        # Create multiple root timers
        stack.start("root1")
        stack.end("root1")
        stack.start("root2")
        stack.end("root2")
        stack.start("root3")
        stack.end("root3")

        # Should only keep max_length root timers
        assert len(stack.root_timers) == 2
        # Oldest root timer should be removed
        root_names = [timer.name for timer in stack.root_timers.items()]
        assert "root1" not in root_names
        assert "root2" in root_names
        assert "root3" in root_names

        # Active stack should not be affected
        assert len(stack.active_stack) == 0


class TestTimerContext:
    def test_timer_context_creation(self):
        ctx = TimerContext(name="test", start_time=1.0)
        assert ctx.name == "test"
        assert ctx.start_time == 1.0
        assert ctx.end_time is None
        assert ctx.duration is None
        assert not ctx.is_complete()

    def test_timer_context_completion(self):
        ctx = TimerContext(name="test", start_time=1.0)
        ctx.end_time = 2.0

        assert ctx.is_complete()
        assert ctx.duration == 1.0
        assert ctx.self_duration == 1.0

    def test_timer_context_with_children(self):
        parent = TimerContext(name="parent", start_time=1.0)
        child = TimerContext(name="child", start_time=1.2, end_time=1.8)
        parent.children.append(child)
        parent.end_time = 2.0

        assert parent.duration == 1.0
        assert parent.self_duration == pytest.approx(0.4)  # 1.0 - 0.6 (child duration)

    def test_to_dict(self):
        ctx = TimerContext(name="test", start_time=1.0, end_time=2.0)
        result = ctx.to_dict()

        assert result["name"] == "test"
        assert result["duration"] == 1.0
        assert result["self_duration"] == 1.0
        assert result["start_time"] == 1.0
        assert result["end_time"] == 2.0
        assert result["children"] == []


class TestTimerStack:
    def test_start_root_timer(self):
        # Reset global state first
        import timingstack as ts

        ts._config = None

        stack = TimerStack()
        ctx = stack.start("root")

        assert ctx.name == "root"
        assert ctx.parent is None
        assert len(stack.root_timers) == 1
        assert len(stack.active_stack) == 1

    def test_start_nested_timer(self):
        # Reset global state first
        import timingstack as ts

        ts._config = None

        stack = TimerStack()
        parent = stack.start("parent")
        child = stack.start("child")

        assert child.parent == parent
        assert len(stack.active_stack) == 2
        assert len(parent.children) == 1

    def test_end_most_recent_timer(self):
        # Reset global state first
        import timingstack as ts

        ts._config = None

        stack = TimerStack()
        stack.start("timer1")
        stack.start("timer2")

        result = stack.end()

        assert result is not None
        assert result.name == "timer2"
        assert result.end_time is not None
        assert len(stack.active_stack) == 1

    def test_end_named_timer(self):
        # Reset global state and configure with reasonable max length
        import timingstack as ts

        ts._config = None
        configure(max_length=1000)

        stack = TimerStack()
        stack.start("timer1")
        stack.start("timer2")
        stack.start("timer3")

        result = stack.end("timer2")

        assert result is not None
        assert result.name == "timer2"
        assert len(stack.active_stack) == 1  # Only timer1 remains

    def test_end_timer_closes_orphans(self):
        # Reset global state and configure with reasonable max length
        import timingstack as ts

        ts._config = None
        configure(max_length=1000)

        stack = TimerStack()
        stack.start("parent")
        stack.start("child1")
        stack.start("child2")

        parent = stack.end("parent")

        assert parent is not None
        assert len(stack.active_stack) == 0
        # All children should be auto-closed
        for child in parent.children:
            assert child.end_time is not None

    def test_close_all(self):
        # Reset global state first
        import timingstack as ts

        ts._config = None

        stack = TimerStack()
        stack.start("timer1")
        stack.start("timer2")

        stack.close_all()

        assert len(stack.active_stack) == 0
        # All timers should have end_time
        for timer in stack.root_timers:
            assert timer.end_time is not None

    def test_get_stats(self):
        # Reset global state first
        import timingstack as ts

        ts._config = None

        stack = TimerStack()
        stack.start("root")
        stack.end("root")

        stats = stack.get_stats()

        assert len(stats) == 1
        assert stats[0]["name"] == "root"
        assert stats[0]["duration"] is not None

    def test_timer_counts(self):
        # Reset global state first
        import timingstack as ts

        ts._config = None

        stack = TimerStack()

        # Test counting different timer names
        stack.start("timer1")
        stack.start("timer2")
        stack.start("timer1")  # timer1 called again
        stack.start("timer3")
        stack.start("timer1")  # timer1 called third time

        counts = stack.get_timer_counts()
        assert counts == {"timer1": 3, "timer2": 1, "timer3": 1}

        # End some timers and verify counts don't change
        stack.end("timer3")
        stack.end("timer1")

        counts = stack.get_timer_counts()
        assert counts == {"timer1": 3, "timer2": 1, "timer3": 1}  # Counts are cumulative

        # Reset and verify counts are cleared
        stack.reset()
        counts = stack.get_timer_counts()
        assert counts == {}

    def test_print_report_with_stats(self):
        # Reset global state first
        import timingstack as ts

        ts._config = None

        stack = TimerStack()

        # Create some timers with known durations
        stack.start("fast_timer")
        time.sleep(0.001)  # Very short sleep
        stack.end("fast_timer")

        stack.start("slow_timer")
        time.sleep(0.01)  # Longer sleep
        stack.end("slow_timer")

        stack.start("nested_timer")
        stack.start("child_timer")
        time.sleep(0.002)
        stack.end("child_timer")
        stack.end("nested_timer")

        # Test that print_report doesn't raise exceptions
        # and includes the enhanced features
        try:
            stack.print_report()
        except Exception as e:
            pytest.fail(f"print_report() raised an exception: {e}")

        # Verify timer counts are being tracked
        counts = stack.get_timer_counts()
        assert "fast_timer" in counts
        assert "slow_timer" in counts
        assert "nested_timer" in counts
        assert "child_timer" in counts


class TestTimer:
    def test_context_manager_sync(self):
        with Timer("test"):
            time.sleep(0.01)

        # Should not raise exception

    def test_context_manager_requires_name(self):
        timer = Timer()
        with pytest.raises(RuntimeError, match="name` must be set"):
            with timer:
                pass

    @pytest.mark.asyncio
    async def test_context_manager_async(self):
        async with Timer("async_test"):
            await asyncio.sleep(0.01)

        # Should not raise exception

    def test_decorator_sync(self):
        @Timer.measure
        def test_func():
            time.sleep(0.01)
            return "result"

        result = test_func()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_decorator_async(self):
        @Timer.measure
        async def test_async_func():
            await asyncio.sleep(0.01)
            return "async_result"

        result = await test_async_func()
        assert result == "async_result"

    def test_decorator_with_custom_name(self):
        @Timer("custom_name")
        def test_func():
            return "result"

        result = test_func()
        assert result == "result"

    def test_manual_start_end(self):
        Timer.start("manual_test")
        time.sleep(0.01)
        ctx = Timer.end("manual_test")

        assert ctx is not None
        assert ctx.name == "manual_test"
        assert ctx.duration is not None and ctx.duration > 0

    def test_timer_disabled_returns_none(self):
        original_config = get_config()

        try:
            # Disable timers
            configure(enabled=False)

            # Timer.start should return None
            ctx = Timer.start("disabled_timer")
            assert ctx is None

            # Timer.end should return None
            ctx = Timer.end("disabled_timer")
            assert ctx is None

        finally:
            # Restore original config
            configure(
                enabled=original_config.enabled,
                on_mismatch=original_config.on_mismatch,
                warn_unclosed=original_config.warn_unclosed,
                auto_close_unclosed=original_config.auto_close_unclosed,
                time_unit=original_config.time_unit,
                precision=original_config.precision,
                logger_name=original_config.logger_name,
                log_level=original_config.log_level,
                max_length=original_config.max_length,
            )

    def test_reset(self):
        Timer.start("test1")
        Timer.start("test2")
        Timer.reset()

        # Should be able to start fresh
        Timer.start("test3")
        Timer.end("test3")

    def test_reset_when_disabled(self):
        original_config = get_config()

        try:
            # Disable timers
            configure(enabled=False)

            # Should not raise exception
            Timer.reset()

        finally:
            # Restore original config
            configure(
                enabled=original_config.enabled,
                on_mismatch=original_config.on_mismatch,
                warn_unclosed=original_config.warn_unclosed,
                auto_close_unclosed=original_config.auto_close_unclosed,
                time_unit=original_config.time_unit,
                precision=original_config.precision,
                logger_name=original_config.logger_name,
                log_level=original_config.log_level,
                max_length=original_config.max_length,
            )


class TestErrorHandling:
    def test_end_without_active_timers(self):
        # Test with default config (WARN)
        result = Timer.end("nonexistent")
        assert result is None

    def test_end_nonexistent_timer_with_raise(self):
        original_config = get_config()

        try:
            configure(on_mismatch=ErrorHandling.RAISE)
            Timer.start("existing")
            with pytest.raises(ValueError, match="no matching start found"):
                Timer.end("nonexistent")
        finally:
            # Restore original config
            configure(
                enabled=original_config.enabled,
                on_mismatch=original_config.on_mismatch,
                warn_unclosed=original_config.warn_unclosed,
                auto_close_unclosed=original_config.auto_close_unclosed,
                time_unit=original_config.time_unit,
                precision=original_config.precision,
                logger_name=original_config.logger_name,
                log_level=original_config.log_level,
                max_length=original_config.max_length,
            )

    def test_mismatched_end_order(self):
        stack = TimerStack()
        stack.start("timer1")
        stack.start("timer2")

        # This should work fine - ends timer2
        result = stack.end("timer1")

        # timer2 should be auto-closed
        assert result is not None
        assert result.name == "timer1"
        assert len(stack.active_stack) == 0


class TestConfiguration:
    def test_timer_config_defaults(self):
        config = TimerConfig()

        assert config.enabled is True
        assert config.on_mismatch == ErrorHandling.WARN
        assert config.warn_unclosed is True
        assert config.auto_close_unclosed is True
        assert config.time_unit == "milliseconds"
        assert config.precision == 2
        assert config.max_length == 1000

    def test_get_config_returns_default(self):
        # Reset global state first
        import timingstack as ts

        ts._config = None

        config = get_config()
        assert isinstance(config, TimerConfig)
        assert config.on_mismatch == ErrorHandling.WARN
        assert config.time_unit == "milliseconds"
        assert config.precision == 2

    def test_configure_updates_settings(self):
        # Reset global state first
        import timingstack as ts

        ts._config = None

        # Test configuring multiple settings
        configure(
            enabled=False,
            time_unit="seconds",
            precision=4,
            warn_unclosed=False,
            on_mismatch=ErrorHandling.RAISE,
            max_length=500,
        )

        config = get_config()
        assert config.enabled is False
        assert config.time_unit == "seconds"
        assert config.precision == 4
        assert config.warn_unclosed is False
        assert config.on_mismatch == ErrorHandling.RAISE
        assert config.max_length == 500

        # Test partial update (only one setting)
        configure(precision=1)

        config = get_config()
        assert config.enabled is False  # Should remain unchanged
        assert config.time_unit == "seconds"  # Should remain unchanged
        assert config.precision == 1  # Should be updated
        assert config.warn_unclosed is False  # Should remain unchanged
        assert config.on_mismatch == ErrorHandling.RAISE  # Should remain unchanged
        assert config.max_length == 500  # Should remain unchanged

    def test_configure_thread_safety(self):
        import threading

        # Reset global state first
        import timingstack as ts

        ts._config = None

        results = []
        errors = []

        def configure_timer(index):
            try:
                configure(precision=index)
                config = get_config()
                results.append((index, config.precision))
            except Exception as e:
                errors.append(e)

        # Start multiple threads configuring simultaneously
        threads = []
        for i in range(10):
            t = threading.Thread(target=configure_timer, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0

        # All threads should have gotten a valid config
        assert len(results) == 10

        # Final config should be from one of the threads (we don't know which one)
        final_config = get_config()
        assert isinstance(final_config, TimerConfig)
        assert 0 <= final_config.precision <= 9

    def test_print_report_units(self):
        stack = TimerStack()
        stack.start("test")
        stack.end("test")

        # Test different time units
        original_config = get_config()
        try:
            configure(time_unit="seconds", precision=3)
            stack.print_report()

            configure(time_unit="microseconds", precision=1)
            stack.print_report()
        finally:
            # Restore original config
            configure(
                enabled=original_config.enabled,
                on_mismatch=original_config.on_mismatch,
                warn_unclosed=original_config.warn_unclosed,
                auto_close_unclosed=original_config.auto_close_unclosed,
                time_unit=original_config.time_unit,
                precision=original_config.precision,
                logger_name=original_config.logger_name,
                log_level=original_config.log_level,
                max_length=original_config.max_length,
            )

    def test_print_report_when_disabled(self):
        original_config = get_config()

        try:
            # Disable timers
            configure(enabled=False)

            # Should print disable message instead of report
            # We'll capture stdout to verify the message
            import io
            import sys

            captured_output = io.StringIO()
            sys_stdout = sys.stdout
            sys.stdout = captured_output

            try:
                Timer.print_report()
                output = captured_output.getvalue()
                assert "TimingStack is currently disabled" in output
                assert "Enable with configure(enabled=True)" in output
            finally:
                sys.stdout = sys_stdout

        finally:
            # Restore original config
            configure(
                enabled=original_config.enabled,
                on_mismatch=original_config.on_mismatch,
                warn_unclosed=original_config.warn_unclosed,
                auto_close_unclosed=original_config.auto_close_unclosed,
                time_unit=original_config.time_unit,
                precision=original_config.precision,
                logger_name=original_config.logger_name,
                log_level=original_config.log_level,
                max_length=original_config.max_length,
            )


if __name__ == "__main__":
    pytest.main([__file__])
