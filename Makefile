.PHONY: install lint test

install:
	python3 -m pip install --upgrade pip && python3 -m pip install -r requirements/requirements.txt
	@echo "Installation complete. You can now run the project."

lint:
	pylint --disable-R,C src/*.py tests/*.py
	@echo "Linting complete. No issues found."

test:
	python -m pytest -vv --cov=src tests/
	@echo "Testing complete. All tests passed."