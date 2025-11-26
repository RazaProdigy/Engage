"""
Demo script showing observability features in action.
Run this to see metrics being collected in real-time.
"""
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.observability import (
    record_llm_call,
    record_retrieval,
    record_entity_extraction,
    record_error,
    get_metrics_text,
    health_checker,
    set_system_info
)


def demo_metrics():
    """Demonstrate various metrics being recorded."""
    
    print("\n" + "="*70)
    print("🔭 Observability Demo - Recording Metrics")
    print("="*70 + "\n")
    
    # Set system info
    set_system_info(version='1.0.0-demo', environment='demo')
    print("✅ System info set\n")
    
    # Simulate LLM calls
    print("📞 Simulating LLM calls...")
    for i in range(3):
        record_llm_call(
            agent='query_understanding',
            model='gpt-4',
            duration=1.5 + i * 0.5,
            prompt_tokens=100 + i * 50,
            completion_tokens=50 + i * 25,
            success=True
        )
        health_checker.record_success()
        time.sleep(0.5)
    print(f"   Recorded 3 successful LLM calls\n")
    
    # Simulate failed call
    record_llm_call(
        agent='response_generation',
        model='gpt-4',
        duration=0.5,
        success=False
    )
    health_checker.record_error()
    print("   Recorded 1 failed LLM call\n")
    
    # Simulate retrieval operations
    print("🔍 Simulating retrieval operations...")
    record_retrieval('hybrid', 0.3, 15, success=True)
    record_retrieval('semantic', 0.2, 12, success=True)
    record_retrieval('bm25', 0.15, 10, success=True)
    print("   Recorded 3 retrieval operations\n")
    
    # Simulate entity extraction
    print("🏷️  Simulating entity extraction...")
    record_entity_extraction(0.8, 4, success=True, fallback=False)
    record_entity_extraction(0.5, 2, success=True, fallback=True)
    print("   Recorded 2 entity extractions (1 with fallback)\n")
    
    # Simulate errors
    print("❌ Simulating errors...")
    record_error('retrieval', 'TimeoutError')
    record_error('query_understanding', 'JSONDecodeError')
    print("   Recorded 2 errors\n")
    
    # Get health stats
    print("="*70)
    print("📊 Health Stats")
    print("="*70)
    stats = health_checker.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    # Print metrics
    print("\n" + "="*70)
    print("📈 Prometheus Metrics Sample")
    print("="*70 + "\n")
    
    metrics = get_metrics_text()
    
    # Print first 50 lines of metrics
    lines = metrics.split('\n')
    for line in lines[:50]:
        if line and not line.startswith('#'):
            print(f"   {line}")
    
    print(f"\n   ... ({len(lines)} total metric lines)")
    
    print("\n" + "="*70)
    print("✅ Demo Complete!")
    print("="*70)
    print("\nIn production, these metrics would be:")
    print("  1. Exposed at http://localhost:8000/metrics")
    print("  2. Scraped by Prometheus every 15s")
    print("  3. Visualized in Grafana dashboards")
    print("  4. Used for alerting on anomalies\n")


if __name__ == "__main__":
    demo_metrics()

