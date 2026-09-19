.PHONY: install test run docker-build docker-test docker-run clean

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest -q

run:
	python src/main.py

docker-build:
	docker build -t ai-jobs-project .

docker-test:
	docker run --rm ai-jobs-project python -m pytest -q

docker-run:
	docker run --rm ai-jobs-project python src/main.py

clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache