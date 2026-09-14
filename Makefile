.PHONY: install generate-data validate build-statements provisions working-capital train simulate optimize test dashboard all clean

PYTHON := python3

install:
	$(PYTHON) -m pip install -e ".[dev]"

generate-data:
	$(PYTHON) -m src.data.generate_data

validate:
	$(PYTHON) -m src.quality.quality_score

build-statements:
	$(PYTHON) -m src.accounting.income_statement
	$(PYTHON) -m src.accounting.balance_sheet
	$(PYTHON) -m src.accounting.cash_flow

provisions:
	$(PYTHON) -m src.provisions.accrual_model

working-capital:
	$(PYTHON) -m src.accounting.working_capital

train:
	$(PYTHON) -m src.forecasting.evaluation

simulate:
	$(PYTHON) -m src.simulation.monte_carlo

optimize:
	$(PYTHON) -m src.optimization.cash_management

test:
	$(PYTHON) -m pytest -v

dashboard:
	streamlit run app/app.py

all: generate-data validate build-statements provisions working-capital train simulate optimize test

clean:
	rm -rf data/raw/*.csv data/processed/*.csv data/processed/*.parquet data/quarantine/*.csv models/*.pkl models/*.joblib reports/figures/* reports/outputs/*
