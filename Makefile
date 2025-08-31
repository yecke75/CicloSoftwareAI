install:
	python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt
	@echo "Installation complete. You can now run the project."

lint:
	pylint --disable-R,C *.py
	@echo "Linting complete. No issues found."

test:
	python -m pytest -vv --cov=res tests/*.py
	@echo "Testing complete. All tests passed."