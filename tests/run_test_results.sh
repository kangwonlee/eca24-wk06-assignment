if grep -q "pytest-xdist" requirements.txt; then
    python -m pytest -n auto tests/test_results.py
else
    python -m pytest tests/test_results.py
fi
