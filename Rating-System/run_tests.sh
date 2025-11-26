#!/bin/bash
# Shell script to run tests for Restaurant Rating Prediction API (Linux/Mac)
# Usage: ./run_tests.sh [option]
# Options: all, unit, api, manual, coverage, help

echo "========================================"
echo "Restaurant Rating API - Test Runner"
echo "========================================"
echo ""

show_help() {
    echo "Usage: ./run_tests.sh [option]"
    echo ""
    echo "Options:"
    echo "  all        - Run all pytest tests (default)"
    echo "  unit       - Run only unit tests for helper functions"
    echo "  api        - Run only API endpoint tests"
    echo "  manual     - Run manual testing script"
    echo "  coverage   - Run tests with coverage report"
    echo "  parallel   - Run tests in parallel (faster)"
    echo "  help       - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_tests.sh              (runs all tests)"
    echo "  ./run_tests.sh coverage     (runs with coverage)"
    echo "  ./run_tests.sh manual       (runs manual tests)"
}

run_all_tests() {
    echo "Running all pytest tests..."
    echo ""
    pytest test_main.py -v
}

run_unit_tests() {
    echo "Running unit tests (helper functions)..."
    echo ""
    pytest test_main.py::TestParsePriceRange test_main.py::TestBuildFeatureDataframe -v
}

run_api_tests() {
    echo "Running API endpoint tests..."
    echo ""
    pytest test_main.py::TestHealthEndpoint test_main.py::TestPredictEndpoint test_main.py::TestRootEndpoint -v
}

run_manual_tests() {
    echo "Running manual test script..."
    echo ""
    echo "Make sure the API server is running!"
    echo "Start server with: uvicorn main:app --reload"
    echo ""
    sleep 2
    python test_api_manual.py
}

run_coverage_tests() {
    echo "Running tests with coverage report..."
    echo ""
    pytest test_main.py -v --cov=main --cov-report=html --cov-report=term
    echo ""
    echo "Coverage report generated in htmlcov/index.html"
    
    # Try to open the coverage report in browser
    if command -v xdg-open > /dev/null; then
        xdg-open htmlcov/index.html
    elif command -v open > /dev/null; then
        open htmlcov/index.html
    else
        echo "Please open htmlcov/index.html in your browser"
    fi
}

run_parallel_tests() {
    echo "Running tests in parallel..."
    echo ""
    if ! command -v pytest-xdist &> /dev/null; then
        echo "Installing pytest-xdist for parallel execution..."
        pip install pytest-xdist
    fi
    pytest test_main.py -v -n auto
}

# Main script logic
case "${1:-all}" in
    all)
        run_all_tests
        ;;
    unit)
        run_unit_tests
        ;;
    api)
        run_api_tests
        ;;
    manual)
        run_manual_tests
        ;;
    coverage)
        run_coverage_tests
        ;;
    parallel)
        run_parallel_tests
        ;;
    help)
        show_help
        exit 0
        ;;
    *)
        echo "Unknown option: $1"
        echo ""
        show_help
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Test run complete!"
echo "========================================"

