@echo off
REM Batch script to run tests for Restaurant Rating Prediction API (Windows)
REM Usage: run_tests.bat [option]
REM Options: all, unit, api, manual, coverage, help

echo ========================================
echo Restaurant Rating API - Test Runner
echo ========================================
echo.

if "%1"=="" goto all_tests
if "%1"=="all" goto all_tests
if "%1"=="unit" goto unit_tests
if "%1"=="api" goto api_tests
if "%1"=="manual" goto manual_tests
if "%1"=="coverage" goto coverage_tests
if "%1"=="help" goto show_help
goto show_help

:all_tests
echo Running all pytest tests...
echo.
pytest test_main.py -v
goto end

:unit_tests
echo Running unit tests (helper functions)...
echo.
pytest test_main.py::TestParsePriceRange test_main.py::TestBuildFeatureDataframe -v
goto end

:api_tests
echo Running API endpoint tests...
echo.
pytest test_main.py::TestHealthEndpoint test_main.py::TestPredictEndpoint test_main.py::TestRootEndpoint -v
goto end

:manual_tests
echo Running manual test script...
echo.
echo Make sure the API server is running!
echo Start server with: uvicorn main:app --reload
echo.
timeout /t 3
python test_api_manual.py
goto end

:coverage_tests
echo Running tests with coverage report...
echo.
pytest test_main.py -v --cov=main --cov-report=html --cov-report=term
echo.
echo Coverage report generated in htmlcov/index.html
start htmlcov\index.html
goto end

:show_help
echo Usage: run_tests.bat [option]
echo.
echo Options:
echo   all        - Run all pytest tests (default)
echo   unit       - Run only unit tests for helper functions
echo   api        - Run only API endpoint tests
echo   manual     - Run manual testing script
echo   coverage   - Run tests with coverage report
echo   help       - Show this help message
echo.
echo Examples:
echo   run_tests.bat              (runs all tests)
echo   run_tests.bat coverage     (runs with coverage)
echo   run_tests.bat manual       (runs manual tests)
goto end

:end
echo.
echo ========================================
echo Test run complete!
echo ========================================

