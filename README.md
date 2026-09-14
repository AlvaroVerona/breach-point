# Breach Point

**Financial Analytics Platform**

An end-to-end financial analytics and decision-intelligence platform: it
validates messy financial data, builds reconciled financial statements,
forecasts revenue/expenses/cash flow, quantifies liquidity risk with Monte
Carlo simulation, and optimizes cash-management decisions to keep the company
above its minimum liquidity requirement.

> Status: project scaffolding in progress. This README will be filled in
> with architecture, methodology and actual results as each phase is built —
> no fabricated numbers.

## Roadmap

- [x] Phase 1 — project setup
- [x] Phase 2 — synthetic data generation
- [x] Phase 3 — data quality engine
- [x] Phase 4 — financial statements
- [x] Phase 5 — provisions / accruals
- [x] Phase 6 — working capital
- [x] Phase 7 — forecasting
- [ ] Phase 8 — Monte Carlo simulation
- [ ] Phase 9 — optimization
- [ ] Phase 10 — Streamlit dashboard
- [ ] Phase 11 — tests
- [ ] Phase 12 — final audit

## Quickstart

```bash
make install
make generate-data
make validate
make test
```

## Synthetic data (Phase 2), actual output from `make generate-data` with seed=42

- 3 entities (EUR/USD/GBP), 107 GL accounts, 45 vendors, 180 customers, 36 months (2023-01 to 2025-12)
- `transactions.csv`: 147,396 double-entry journal lines (73,097 documents)
- `accounts_receivable.csv`: 18,188 invoices — `accounts_payable.csv`: 19,498 invoices
- Revenue and expenses are generated first as monthly targets (trend + seasonality +
  business-unit mix + noise, with a deliberate revenue slowdown in the final months); GL
  transactions are then generated to sum to those targets, so financial statements built
  from the ledger stay consistent with the series forecasting will later be trained on.
- Balance Sheet balances exactly on the 3 opening entries; of the 73,097 posted documents,
  1,307 (~1.8%) are intentionally unbalanced — partly the deliberately injected broken
  entries, partly an emergent effect of exact-duplicate row injection (a duplicated single
  leg breaks that document's balance too, which is realistic).
- Injected issues (rates in `config/settings.yaml`): ~0.8% exact/duplicate-invoice
  duplicates, missing values (differentiated by field — 5% on optional metadata, 1% on
  account_id, 0.05% on amount), invalid currencies/accounts/negative amounts, inconsistent
  account-category labeling, unbalanced journal entries, one dropped entity-month of AP data,
  and a handful of syntactically invalid date strings.
- Reproducible: identical seed produces a byte-identical `transactions.csv` (see
  `tests/test_data_generation.py::test_generate_all_reproducible`).

## Data quality engine (Phase 3), actual output from `make validate`

- **Financial Data Quality Score: 97.9 / 100** — Completeness 97.3, Uniqueness 98.9,
  Validity 98.4, Consistency 99.6, Reconciliation 95.4 (weighted average; every number
  computed from the actual check results, see `reports/outputs/quality_report.json`).
- ~15,000 records (of 185,082 across the three row-level datasets) quarantined for a
  CRITICAL or HIGH-severity issue — MEDIUM/LOW/INFO issues (e.g. statistical outliers,
  unpaid-invoice missing payment dates) stay in the validated layer.
  Quarantine cascades to the whole journal document when any one leg is flagged, so the
  validated `transactions.csv` has **zero unbalanced documents** — verified by
  `test_validated_transactions_have_no_unbalanced_documents`, and the reason the
  reconciliation score is a few points lower than the other dimensions: a document with
  one flagged leg costs the whole document, not just that leg. Opening-balance journal
  entries (one per entity, seeding the whole Balance Sheet) are excluded from injection
  entirely — losing one would cascade to the entity's entire opening position, which is a
  realistic failure mode but not an interesting one to demonstrate by accident.

## Financial statements (Phase 4), actual output from `make build-statements`

- Income Statement, Balance Sheet and Cash Flow built monthly per entity, in EUR, from
  `data/processed/` (the VALIDATED layer) — 108 entity-months (3 entities × 36 months) per
  statement.
- **Balance Sheet reconciles 108/108 entity-months** (Assets = Liabilities + Equity, by
  double-entry construction — see CLAUDE.md) and **Cash Flow reconciles 108/108**
  (independently computed Operating/Investing/Financing CF ties to the Balance Sheet's own
  cash balance, not derived from it — a genuine check, not a tautology).
- Corporate income tax isn't journaled in the ledger (Phase 2 scope); it's estimated at
  statement-build time as a flat statutory rate on pre-tax income, with an explicit
  `income_tax_payable` accrual added to the Balance Sheet so the notional tax deduction
  doesn't break the accounting equation — this was a real bug caught during this phase (the
  Balance Sheet failed to reconcile on 108/108 entity-months before the fix, drifting
  further out of balance every month by exactly the cumulative tax amount).
- Financing CF is 0 historically — the ledger has no debt issuance/repayment transactions,
  only interest on a constant opening balance; new borrowing is a Phase 9 optimization
  lever, not part of these actuals.
- **Known, transparently-flagged limitation**: 9 of ENT_UK's 36 entity-months have a
  negative Inventory balance (`economically_implausible = True` in `balance_sheet.csv`).
  The Balance Sheet still reconciles exactly — this is quarantine removing more
  purchase-side inventory records (which carry more fields, so more exposure to flagged
  issues) than consumption-side ones for the entity with the smallest transaction volume,
  not a data-generation error (raw, pre-quarantine data shows healthy positive inventory
  for ENT_UK throughout). Surfaced explicitly rather than silently floored.

## Provisions / accruals (Phase 5), actual output from `make provisions`

- `Accrual_t = ExpectedExpense_t - RecognizedExpense_t` for Utilities, Professional Services
  and Logistics (per entity/business unit) and Interest Expense (per entity) — 1,619
  monthly estimates, `expected_cost` from a trailing 3-month rolling average (with the
  expanding-window historical average and same-calendar-month seasonal average also
  reported, for transparency on the method choice), plus a 95% confidence interval from the
  standard error of that rolling window.
- The estimator is **not systematically biased**: mean accrual per category is small
  relative to mean recognized cost — Utilities +€3.4 vs. €1,087 average, Professional
  Services -€7.1 vs. €1,545, Logistics -€29.2 vs. €4,435, Interest Expense +€19.8 vs.
  €9,131 — verified by `test_estimator_is_not_systematically_biased` (mean accrual < 15% of
  mean recognized cost for every category). This dataset has no artificial invoice-arrival
  lag (every AP invoice is dated within the month it belongs to), so these accruals reflect
  genuine month-to-month estimation variance in a recurring cost, not a fabricated
  reporting gap.

## Working capital (Phase 6), actual output from `make working-capital`

- DSO, DPO, DIO and Cash Conversion Cycle per entity-month, built purely as ratios of the
  Phase 4 statements — no new data. DPO deliberately uses `AP / (COGS + Operating Expense)`
  instead of the textbook `AP / COGS`: this company's AP funds a broad vendor base (rent,
  software, marketing, logistics, professional services, utilities — not just inventory
  purchases), so a COGS-only denominator inflated DPO to 150-250+ days; documented in the
  module docstring and CLAUDE.md.
- Ranges across the 36-month history: DSO 56-108 days, DPO 56-132 days, DIO 22-215 days,
  CCC mostly positive (12-165 days, with ENT_EU briefly dipping to -4).
- **Two real Phase 2 bugs found and fixed while building this**, both in
  `src/data/transaction_events.py`: (1) no bad-debt write-off mechanism meant ~7% of AR
  invoices were structurally stuck unpaid forever, making DSO grow without bound over the
  36-month window (985 invoices were already >6 months overdue with no resolution) — fixed
  by writing off invoices 120 days past due to a real `Bad Debt Expense` entry, an account
  the chart of accounts had reserved since Phase 2 but never used; (2) tuned the "genuinely
  stuck" probability down on both AR (7%→3%) and AP (5%→1%) — AP has no write-off
  equivalent (a going concern eventually pays its vendors), so a high stuck-rate there would
  have left AP growing unboundedly the same way.

## Forecasting (Phase 7), actual output from `make train`

- 12-month-ahead forecasts for Revenue, Operating Expense, Accounts Receivable, Accounts
  Payable, Operating Cash Flow and Ending Cash, per entity (18 series total) — baseline
  (naive, seasonal naive), statistical (Holt-Winters Exponential Smoothing) and ML
  (scikit-learn `GradientBoostingRegressor`, recursive multi-step, lag/rolling/cyclical
  features) all evaluated, with the lowest-RMSE model on a genuine 6-month blind holdout
  selected per series and refit on full history for the real forecast.
- Model selection was close to a 4-way split (naive 7, gradient boosting 4, exponential
  smoothing 4, seasonal naive 3, out of 18 series) — no single family dominates, which is
  itself a useful signal given only 36 months of history per entity.
- 95% confidence intervals come from the selected model's own held-out validation residuals
  (widening with √step under a random-walk-error assumption), not from in-sample fit —
  verified widening by `test_confidence_interval_widens_with_horizon`.
- **Naive wins for Revenue in all three entities**, and it's a real, explainable finding,
  not a fluke: the 6-month validation window sits entirely inside Phase 2's deliberate
  late-window revenue slowdown, where a trend/seasonal model extrapolating the prior growth
  pattern overshoots right as the regime shifts, while a flat "no change" forecast does
  comparatively less damage. Exactly the kind of instability Phase 8's Monte Carlo exists to
  quantify, not paper over.
- MAPE on Operating Cash Flow is unstable (up to ~280% for one entity) because that series
  crosses close to zero some months — a known MAPE limitation (dividing by a near-zero
  actual), not a forecasting failure; RMSE and sMAPE (also reported) are the more reliable
  metrics for that series.
- Leakage prevention (§45): every validation-period forecast is produced from the train
  prefix only — verified by `test_no_leakage_validation_forecast_depends_only_on_train`
  (corrupting the held-out values doesn't change the forecast, because the forecasting
  function is never given access to them in the first place).
