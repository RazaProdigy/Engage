#!/usr/bin/env python3
"""
Simple test script to verify latency logging functionality.
Run this after making any changes to the observability module.
"""
import time
from pathlib import Path

# Import functions from observability
from src.observability import (
    record_llm_call,
    record_retrieval,
    record_entity_extraction,
    get_latest_latency_metrics,
    get_latency_summary,
    LATENCY_LOG_FILE
)


def test_latency_logging():
    """Test that latency metrics are being logged correctly."""
    
    print("Testing latency logging functionality...\n")
    
    # Clear any existing log file for clean test
    if LATENCY_LOG_FILE.exists():
        LATENCY_LOG_FILE.unlink()
        print("✓ Cleared existing log file")
    
    # Test 1: Record LLM call
    print("\n1. Testing LLM call logging...")
    record_llm_call(
        agent='test_agent',
        model='gpt-4',
        duration=1.234,
        prompt_tokens=100,
        completion_tokens=50,
        success=True
    )
    print("✓ Recorded LLM call")
    
    # Small delay to ensure different timestamps
    time.sleep(0.1)
    
    # Test 2: Record retrieval
    print("\n2. Testing retrieval logging...")
    record_retrieval(
        retriever_type='hybrid',
        duration=0.234,
        num_documents=10,
        success=True
    )
    print("✓ Recorded retrieval")
    
    time.sleep(0.1)
    
    # Test 3: Record entity extraction
    print("\n3. Testing entity extraction logging...")
    record_entity_extraction(
        duration=0.567,
        num_entities=3,
        success=True,
        fallback=False
    )
    print("✓ Recorded entity extraction")
    
    time.sleep(0.1)
    
    # Test 4: Record failed LLM call
    print("\n4. Testing failed LLM call logging...")
    record_llm_call(
        agent='test_agent',
        model='gpt-4',
        duration=0.5,
        success=False
    )
    print("✓ Recorded failed LLM call")
    
    # Test 5: Verify log file exists
    print("\n5. Verifying log file...")
    if LATENCY_LOG_FILE.exists():
        size_bytes = LATENCY_LOG_FILE.stat().st_size
        print(f"✓ Log file exists ({size_bytes} bytes)")
    else:
        print("✗ Log file does not exist!")
        return False
    
    # Test 6: Read latest metrics
    print("\n6. Testing get_latest_latency_metrics...")
    latest = get_latest_latency_metrics(num_lines=10)
    if "llm_call" in latest and "retrieval" in latest:
        print("✓ Successfully retrieved latest metrics")
        print("\nLatest metrics:")
        print("-" * 80)
        print(latest)
    else:
        print("✗ Failed to retrieve metrics!")
        return False
    
    # Test 7: Get summary statistics
    print("\n7. Testing get_latency_summary...")
    summary = get_latency_summary()
    if summary and "llm_call" in summary and "retrieval" in summary:
        print("✓ Successfully calculated summary statistics")
        print("\nSummary:")
        print("-" * 80)
        for metric_type, stats in summary.items():
            print(f"{metric_type:25s} | Count: {stats['count']:3d} | "
                  f"Avg: {stats['avg_ms']:8.2f}ms | Min: {stats['min_ms']:8.2f}ms | "
                  f"Max: {stats['max_ms']:8.2f}ms")
    else:
        print("✗ Failed to calculate summary!")
        return False
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    print(f"\nLog file location: {LATENCY_LOG_FILE.absolute()}")
    print("\nYou can view metrics with:")
    print("  python view_metrics.py")
    print("  python view_metrics.py --summary")
    
    return True


if __name__ == "__main__":
    try:
        success = test_latency_logging()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

