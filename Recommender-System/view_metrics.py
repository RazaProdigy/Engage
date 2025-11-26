#!/usr/bin/env python3
"""
Quick script to view latency metrics from the log file.
Usage:
    python view_metrics.py              # View last 50 entries
    python view_metrics.py --lines 100  # View last 100 entries
    python view_metrics.py --summary    # View summary statistics
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.observability import (
    get_latest_latency_metrics,
    get_latency_summary,
    LATENCY_LOG_FILE
)


def main():
    parser = argparse.ArgumentParser(
        description="View latency metrics for retrieval and LLM calls"
    )
    parser.add_argument(
        '--lines', '-n',
        type=int,
        default=50,
        help="Number of recent log entries to display (default: 50)"
    )
    parser.add_argument(
        '--summary', '-s',
        action='store_true',
        help="Show summary statistics instead of recent entries"
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help="Show all entries"
    )
    parser.add_argument(
        '--file', '-f',
        action='store_true',
        help="Show the log file path"
    )
    
    args = parser.parse_args()
    
    # Show file path
    if args.file:
        print(f"Latency log file: {LATENCY_LOG_FILE.absolute()}")
        if LATENCY_LOG_FILE.exists():
            size_kb = LATENCY_LOG_FILE.stat().st_size / 1024
            print(f"File size: {size_kb:.2f} KB")
        else:
            print("File does not exist yet (no metrics logged)")
        return
    
    # Show summary
    if args.summary:
        print("=== Latency Metrics Summary ===\n")
        summary = get_latency_summary()
        
        if "message" in summary or "error" in summary:
            print(summary.get("message") or summary.get("error"))
            return
        
        if not summary:
            print("No metrics found.")
            return
        
        # Print table header
        print(f"{'Metric Type':<25} {'Count':>8} {'Avg (ms)':>12} {'Min (ms)':>12} {'Max (ms)':>12} {'Latest (ms)':>12}")
        print("-" * 90)
        
        # Print each metric type
        for metric_type, stats in sorted(summary.items()):
            print(f"{metric_type:<25} {stats['count']:>8} "
                  f"{stats['avg_ms']:>12.2f} {stats['min_ms']:>12.2f} "
                  f"{stats['max_ms']:>12.2f} {stats['latest_ms']:>12.2f}")
        
        return
    
    # Show recent entries
    num_lines = None if args.all else args.lines
    metrics = get_latest_latency_metrics(num_lines or 1000000)
    print(metrics)


if __name__ == "__main__":
    main()

