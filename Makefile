.PHONY: install test lint verify run run-all demo

install:
	python -m pip install -r requirements-paper.txt
	python -m pip install -e .

test:
	pytest -q

lint:
	ruff check src tests scripts app.py

verify:
	python scripts/verify_repository.py
	python scripts/verify_notebook_contract.py

run:
	python -m life_satisfaction.pipeline --data data/raw/SHILD2012_cens_eng_16-64.dta --config configs/paper.yaml --output results/generated

run-all:
	python -m life_satisfaction.pipeline --data data/raw/SHILD2012_cens_eng_16-64.dta --config configs/paper.yaml --output results/generated --all-analyses

demo:
	life-satisfaction-demo --model results/generated/best_model.pkl
