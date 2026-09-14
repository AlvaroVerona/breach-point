# Breach Point — engineering notes for Claude Code

Financial Data Quality, Forecasting & Liquidity Optimization Platform.
Portfolio project: raw, imperfect financial data → validated data → financial
statements → forecasts → Monte Carlo liquidity risk → optimized cash
management → Streamlit dashboard.

## Key decisions (do not relitigate without asking)

- **ML forecasting model**: scikit-learn `GradientBoostingRegressor` only.
  No XGBoost dependency — kept the stack minimal on purpose.
- **Monte Carlo ↔ Optimization link**: optimize cash management against the
  *base forecast* (deterministic LP/CP via OR-Tools), then re-run Monte Carlo
  on the optimized strategy to report the before/after liquidity breach
  probability. Not a full chance-constrained stochastic program — kept simple
  and interpretable per the project's own engineering principles.
- **Reconciliation applies to the VALIDATED layer, not RAW.** Raw data
  intentionally contains unbalanced journal entries and broken accounting
  identities (injected for the quality engine to catch). Balance Sheet /
  Income Statement / Cash Flow reconciliation checks (and the "PASS" status
  in the definition of done) run on `data/processed/` (post-quarantine), not
  on `data/raw/`.
- **Journal-entry balance vs. statement-level balance are two different
  checks.** `sum(debit) == sum(credit)` per `document_id` is a transaction-
  level quality check (`src/quality/reconciliation.py`). `Assets = Liabilities
  + Equity` is a statement-level accounting check (`src/accounting/`). Don't
  conflate them.
- **Revenue/expense generative process is the source of truth.** Monthly
  revenue and expense series (with seasonality/trend/business-unit
  variation) are generated first; GL transactions in `transactions.csv` are
  then generated consistent with those monthly totals. This keeps the
  financial statements built from transactions consistent with the series
  used for forecasting — they are not two independent random processes.
- **Row-level lineage key.** Every raw record gets a synthetic `row_uid`
  (independent of business keys like `transaction_id`) at ingestion, before
  any validation. This is what quarantine references when a business-key
  field itself is null.
- **FX rates**: generated as `data/raw/fx_rates.csv` (monthly, per currency
  pair, seeded), not hardcoded. Base currency is EUR.
- **Master data**: `data/raw/entities.csv`, `chart_of_accounts.csv`,
  `vendors.csv`, `customers.csv` are generated explicitly — referential
  integrity checks validate against these, not against values inferred from
  the transaction data itself.

## Engineering principles

Business logic lives in `src/`, never in notebooks. All stochastic code
takes its seed from `config/settings.yaml` (default 42) — nothing hardcodes
a seed inline. No print statements for anything that matters; use
`src.common.logging_config.get_logger(__name__)`. Never fabricate a result —
every number in the README/dashboard must come from an actual pipeline run.

## Workflow

Work in phases (data generation → quality engine → statements → provisions →
working capital → forecasting → Monte Carlo → optimization → dashboard →
tests → final audit). After each phase, run the relevant tests and inspect
generated output before moving to the next one.
