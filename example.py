"""
Comprehensive examples demonstrating the timingstack library features.
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from timingstack import (
    ErrorHandling,
    Timer,
    TimerStack,
    configure,
    get_config,
)


def example_1_basic_usage():
    """Example 1: Basic timer usage with different methods"""
    print("=" * 60)
    print("Example 1: Basic Timer Usage")
    print("=" * 60)

    # Method 1: Context manager
    print("\n1. Using Timer as context manager:")
    with Timer("database_connection"):
        time.sleep(0.1)

    # Method 2: Manual start/stop
    print("\n2. Using manual start/stop:")
    Timer.start("api_request")
    time.sleep(0.05)
    Timer.end("api_request")

    # Method 3: Decorator
    print("\n3. Using Timer as decorator:")

    @Timer.measure
    def process_data():
        time.sleep(0.08)
        return "processed"

    result = process_data()
    print(f"   Function returned: {result}")

    # Show the report
    Timer.print_report()


def example_2_async_support():
    """Example 2: Async timer support"""
    print("\n" + "=" * 60)
    print("Example 2: Async Timer Support")
    print("=" * 60)

    # Async context manager
    print("\n1. Using Timer with async context manager:")

    async def fetch_data():
        async with Timer("http_request"):
            await asyncio.sleep(0.1)
        return "data"

    # Async decorator
    print("\n2. Using Timer with async decorator:")

    @Timer.measure
    async def async_operation():
        await asyncio.sleep(0.06)
        return "async_result"

    async def main():
        # Run async operations
        await fetch_data()
        result = await async_operation()
        print(f"   Async function returned: {result}")

    asyncio.run(main())

    # Show the report
    Timer.print_report()


def example_3_nested_timers():
    """Example 3: Nested timers and hierarchy"""
    print("\n" + "=" * 60)
    print("Example 3: Nested Timers and Hierarchy")
    print("=" * 60)

    @Timer.measure
    def process_order():
        print("   Processing order...")

        with Timer("validate_order"):
            time.sleep(0.02)
            print("   ✓ Order validated")

        with Timer("calculate_price"):
            time.sleep(0.015)
            print("   ✓ Price calculated")

        with Timer("save_to_database"):
            time.sleep(0.05)

            with Timer("create_connection"):
                time.sleep(0.01)
                print("   ✓ Database connection created")

            with Timer("execute_query"):
                time.sleep(0.03)
                print("   ✓ Query executed")

            print("   ✓ Order saved to database")

    process_order()

    # Show the report with detailed hierarchy
    Timer.print_report()


def example_4_timer_counts_and_statistics():
    """Example 4: Aggregated timer statistics per timer"""
    print("\n" + "=" * 60)
    print("Example 4: Aggregated Timer Statistics")
    print("=" * 60)

    # Configure for milliseconds display
    configure(time_unit="milliseconds", precision=3)

    print("\nRunning multiple iterations with varying durations...")

    # Run operations multiple times with varying durations
    for i in range(5):
        with Timer("data_processing"):
            time.sleep(0.002 + (i * 0.0005))  # 2ms, 2.5ms, 3ms, 3.5ms, 4ms

    for i in range(3):
        with Timer("file_io"):
            time.sleep(0.005 + (i * 0.001))  # 5ms, 6ms, 7ms

    for i in range(2):
        with Timer("network_request"):
            time.sleep(0.010 + (i * 0.002))  # 10ms, 12ms

    # Nested operations (will be aggregated by name)
    with Timer("batch_process"):
        for i in range(3):
            with Timer("process_item"):
                time.sleep(0.001)
            with Timer("validate_item"):
                time.sleep(0.0005)

    # Single operations
    with Timer("initialization"):
        time.sleep(0.020)

    with Timer("cleanup"):
        time.sleep(0.003)

    # Show the new aggregated report with per-timer statistics
    print("\nAggregated timing report with per-timer statistics:")
    Timer.print_report()


def example_5_error_handling():
    """Example 5: Error handling configuration"""
    print("\n" + "=" * 60)
    print("Example 5: Error Handling Configuration")
    print("=" * 60)

    print("\n1. Default error handling (WARN):")
    try:
        Timer.end("nonexistent_timer")
    except:
        print("   Exception raised unexpectedly!")

    print("\n2. Configure to raise exceptions:")
    configure(on_mismatch=ErrorHandling.RAISE)

    try:
        Timer.end("nonexistent_timer")
    except ValueError as e:
        print(f"   Caught expected exception: {e}")

    # Reset to default
    configure(on_mismatch=ErrorHandling.WARN)

    print("\n3. Configure to ignore errors:")
    configure(on_mismatch=ErrorHandling.IGNORE)

    # This should not raise or warn
    Timer.end("another_nonexistent_timer")
    print("   No warning or exception raised")

    # Reset to default
    configure(on_mismatch=ErrorHandling.WARN)


def example_6_bounded_list_behavior():
    """Example 6: Bounded list and memory management"""
    print("\n" + "=" * 60)
    print("Example 6: Bounded List and Memory Management")
    print("=" * 60)

    # Configure with small max length to demonstrate behavior
    configure(max_length=3, warn_unclosed=True)

    stack = TimerStack()

    print("\nCreating root timers (max_length=3):")

    # Create multiple root timers
    for i in range(5):
        with Timer(f"operation_{i + 1}"):
            time.sleep(0.001)
            print(f"   Completed operation_{i + 1}")

    print("\nTotal root timers created: 5")
    print(f"Root timers kept (due to max_length): {len(stack.root_timers)}")

    kept_names = [timer.name for timer in stack.root_timers.items()]
    print(f"Kept timers: {kept_names}")

    # Show the report
    Timer.print_report()


def example_7_configuration_options():
    """Example 7: Configuration options"""
    print("\n" + "=" * 60)
    print("Example 7: Configuration Options")
    print("=" * 60)

    # Show default configuration
    print("\n1. Default configuration:")
    config = get_config()
    print(f"   Time unit: {config.time_unit}")
    print(f"   Precision: {config.precision}")
    print(f"   Max length: {config.max_length}")
    print(f"   Warn unclosed: {config.warn_unclosed}")
    print(f"   Error handling: {config.on_mismatch.value}")

    print("\n2. Configure for seconds with high precision:")
    configure(time_unit="seconds", precision=6, max_length=100)

    with Timer("precise_timing"):
        time.sleep(0.001234)

    Timer.print_report()

    print("\n3. Configure for microseconds:")
    configure(time_unit="microseconds", precision=0)

    with Timer("microsecond_timing"):
        time.sleep(0.0005)

    Timer.print_report()


def example_8_custom_timer_stack():
    """Example 8: Using TimerStack directly"""
    print("\n" + "=" * 60)
    print("Example 8: Using TimerStack Directly")
    print("=" * 60)

    # Create separate timer stacks for different contexts
    web_stack = TimerStack()
    background_stack = TimerStack()

    print("\n1. Using web stack:")
    web_stack.start("http_request")
    time.sleep(0.02)
    web_stack.start("json_parsing")
    time.sleep(0.005)
    web_stack.end("json_parsing")
    web_stack.end("http_request")

    print("\n2. Using background stack:")
    background_stack.start("file_processing")
    time.sleep(0.03)
    background_stack.start("data_validation")
    time.sleep(0.01)
    background_stack.end("data_validation")
    background_stack.end("file_processing")

    print("\n3. Web stack report:")
    web_stack.print_report()

    print("\n4. Background stack report:")
    background_stack.print_report()

    print("\n5. Timer call counts:")
    print(f"   Web stack: {web_stack.get_timer_counts()}")
    print(f"   Background stack: {background_stack.get_timer_counts()}")


def example_9_advanced_nested_scenarios():
    """Example 9: Advanced nested scenarios with loops"""
    print("\n" + "=" * 60)
    print("Example 9: Advanced Nested Scenarios")
    print("=" * 60)

    @Timer.measure
    def process_batch(items):
        print(f"   Processing batch of {len(items)} items...")

        total_item_time = 0

        for i, item in enumerate(items):
            with Timer("process_item"):
                # Simulate varying processing times
                processing_time = 0.001 + (i * 0.0002)
                time.sleep(processing_time)
                total_item_time += processing_time

                with Timer("validate_item"):
                    time.sleep(0.0002)

                with Timer("save_item"):
                    time.sleep(0.0003)

        print("   Batch processing completed")

    # Process multiple batches
    for batch_id in range(3):
        print(f"\nProcessing batch {batch_id + 1}:")
        items = list(range(5))
        process_batch(items)

    print("\nFinal timing report with comprehensive statistics:")
    Timer.print_report()


def example_10_error_recovery_and_cleanup():
    """Example 10: Error recovery and cleanup scenarios"""
    print("\n" + "=" * 60)
    print("Example 10: Error Recovery and Cleanup")
    print("=" * 60)

    # Configure to warn about unclosed timers
    configure(warn_unclosed=True, auto_close_unclosed=True)

    print("\n1. Simulating unclosed timers:")
    Timer.start("cleanup_test_1")
    Timer.start("cleanup_test_2")

    # Don't close them - they should be auto-closed on reset
    print("   Started timers but didn't close them...")

    print("\n2. Resetting (should auto-close and warn):")
    Timer.reset()

    print("\n3. Timer mismatch scenario:")
    Timer.start("outer")
    Timer.start("inner")

    # End outer directly (should auto-close inner)
    print("   Ending outer timer directly...")
    Timer.end("outer")

    print("\n4. Final report:")
    Timer.print_report()


def example_11_enable_disable_functionality():
    """Example 11: Enable/disable functionality"""
    print("\n" + "=" * 60)
    print("Example 11: Enable/Disable Functionality")
    print("=" * 60)

    print("\n1. Running with timers ENABLED (default):")

    # Enable timers explicitly
    configure(enabled=True)

    with Timer("enabled_operation"):
        time.sleep(0.01)

    Timer.print_report()

    print("\n2. Running with timers DISABLED:")

    # Disable timers
    configure(enabled=False)

    with Timer("disabled_operation"):
        time.sleep(0.01)

    # Try manual timer operations
    Timer.start("manual_disabled")
    time.sleep(0.01)
    Timer.end("manual_disabled")

    print("\n3. Trying to print report when disabled:")
    Timer.print_report()

    print("\n4. Re-enabling timers:")

    # Re-enable timers
    configure(enabled=True)

    with Timer("re_enabled_operation"):
        time.sleep(0.01)

    print("\n5. Final report (should only show enabled operations):")
    Timer.print_report()


def main():
    """Run all examples"""
    print("TimingStack Library - Comprehensive Examples")
    print("=" * 60)
    print("This demo showcases all the features of the timingstack library")
    print("including timer counting, statistics, nested timers, and more.")

    # Reset configuration before examples
    Timer.reset()

    # Run all examples
    example_1_basic_usage()

    Timer.reset()
    example_2_async_support()

    Timer.reset()
    example_3_nested_timers()

    Timer.reset()
    example_4_timer_counts_and_statistics()

    Timer.reset()
    example_5_error_handling()

    Timer.reset()
    example_6_bounded_list_behavior()

    Timer.reset()
    example_7_configuration_options()

    Timer.reset()
    example_8_custom_timer_stack()

    Timer.reset()
    example_9_advanced_nested_scenarios()

    Timer.reset()
    example_10_error_recovery_and_cleanup()

    Timer.reset()
    example_11_enable_disable_functionality()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("✓ Multiple timer usage patterns (context manager, decorator, manual)")
    print("✓ Async/await support")
    print("✓ Nested timer hierarchies")
    print("✓ Timer call counting and statistics")
    print("✓ Configurable error handling")
    print("✓ Memory management with bounded lists")
    print("✓ Flexible configuration options")
    print("✓ Multiple independent timer stacks")
    print("✓ Advanced scenarios with loops and batch processing")
    print("✓ Error recovery and automatic cleanup")
    print("✓ Global enable/disable functionality")


if __name__ == "__main__":
    main()
