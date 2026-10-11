requirements:
	pip install -r requirements.txt

data: requirements
	python -m heart_ccds.dataset

train: data
	python -m heart_ccds.modeling.train

lint:
	ruff check heart_ccds

test:
	pytest -q