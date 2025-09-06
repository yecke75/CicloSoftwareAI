.PHONY: install lint test

install:
	python3 -m pip install --upgrade pip && python3 -m pip install -r requirements/requirements.txt
	@echo "Installation complete. You can now run the project."

lint:
	python -c "import sys; sys.path.insert(0, 'src'); import pylint.lint; pylint.lint.Run(['--disable=R,C,W,E1101', 'src/*.py', 'tests/*.py'])"
	@echo "Linting complete. No issues found."

test:
	python -m pytest -vv --cov=src tests/
	@echo "Testing complete. All tests passed."

build:
	python -m build
	@echo "Build complete. Check the dist/ directory."

clean:
	-rd /s /q dist
	-rd /s /q build
	-rd /s /q .pytest_cache
	-rd /s /q __pycache__
	-rd /s /q src\__pycache__
	-rd /s /q tests\__pycache__
	-rd /s /q src\ciclosoftwareai.egg-info
	@echo "Clean complete."