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
- [x] Phase 8 — Monte Carlo simulation
- [x] Phase 9 — optimization
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

## Monte Carlo & liquidity risk (Phase 8), actual output from `make simulate risk`

- 10,000 simulations × 12 months × 3 entities, consolidated to a single cash-balance
  distribution per month. Every stochastic input is calibrated from real historical/forecast
  data rather than invented: Revenue and Operating Expense noise comes from each series' own
  Phase 7 held-out validation residual std; customer/supplier payment-timing risk is
  simulated DSO/DPO drawn from their Phase 6 historical distributions (so a slower-paying
  customer base shows up directly as a cash effect, not just a generic volatility knob);
  COGS is simulated Revenue × a historical gross-margin distribution; "other" cash flows use
  the historical Investing CF (CAPEX) distribution net of a near-deterministic interest
  outflow. Base/Optimistic/Pessimistic scenarios (§32) shift the simulation's center
  (Revenue ±10%, DSO/DPO shift) before the Monte Carlo explores uncertainty around it.
- **Recalibrated `liquidity.minimum_cash` from the spec's illustrative €1,000,000 example to
  €2,750,000** (~2 months of this company's actual ~€1.37M/month consolidated operating cash
  outflow — a standard treasury buffer policy) — the literal spec number is <1 month of
  spend for a company holding ~€18M cash, which would make every risk score trivially LOW
  regardless of scenario and defeat the point of running the simulation at all.
- **Liquidity Risk: LOW in all three scenarios** — 0% probability of breach, worst simulated
  consolidated cash ~€13.3M even under the pessimistic scenario, comfortably above the
  €2.75M threshold. This is an honest result, not a disappointing one: the synthetic company
  is well-capitalized and profitable, and the Monte Carlo correctly reflects that rather than
  being tuned to manufacture a crisis. `test_pessimistic_breach_probability_not_below_optimistic`
  and `test_scenario_direction_is_correct` confirm the simulation responds in the right
  direction to each scenario even though none of them currently breach.
- Since the base/scenario analysis doesn't produce a liquidity event to react to, Phase 9's
  optimization will additionally test a deliberately more severe stress case, to exercise the
  cash-management decision mechanism (borrowing, collections acceleration, CAPEX deferral)
  under conditions where it actually has something to solve — documented explicitly as a
  stress test, not presented as the base-case forecast.

## Optimization (Phase 9), actual output from `make optimize`

- A continuous LP (OR-Tools GLOP) chooses the lowest-cost combination of short-term
  borrowing, receivables factoring (accelerated collections), payment deferral and CAPEX
  reduction that keeps consolidated cash at or above a minimum-cash requirement in every
  month, against a deterministic baseline (point forecasts, no randomness — consistent with
  the project's documented "optimize against the expected case, then re-run Monte Carlo"
  split).
- **Two calibration problems found and fixed while building this, both documented in
  CLAUDE.md rather than silently patched over:**
  1. The real liquidity policy (€2.75M) is never breached even under the Phase 8 stress
     scenario (€11.3M of headroom remains) — optimizing against it trivially finds "do
     nothing." Rather than manufacture an artificial crisis by inflating the stress
     assumptions to absurd levels (tried up to -95% revenue and still didn't breach the
     *consolidated*, cash-pooled position — a genuinely interesting diversification-benefit
     finding in its own right), the demonstration uses an explicitly-labeled hypothetical
     stricter policy (`optimization.demo_minimum_cash`, €16M) solely to exercise the decision
     mechanism — never presented as the real liquidity policy.
  2. A deterministic LP has no concept of uncertainty: optimizing against the point-forecast
     baseline alone barely helped when re-evaluated against the actual Monte Carlo
     distribution (100% → 97.5% breach probability). Added a volatility-based safety buffer
     (`SAFETY_BUFFER_Z` standard deviations of the Monte Carlo's own month-by-month spread,
     a standard chance-constrained-LP approximation) — a real 0/1/2/3-sigma comparison
     (100% → 71% → 23% → 2.2% breach probability) showed 1 and 2 sigma were genuinely
     under-protective, not just cheaper.
- **Result: Liquidity Risk CRITICAL (100% breach probability) → LOW (2.2%)** under the
  hypothetical €16M demo policy and stress scenario, for a total cost of ~€50,081 (financing
  €16,391 + factoring €33,690; the CAPEX-deferral lever went unused — cheaper levers covered
  the shortfall first, a genuine LP result, not a hardcoded preference).
- Feasibility, the cash constraint, and non-negativity of every decision variable are
  verified on both handcrafted LP fixtures and the real problem
  (`tests/test_optimization.py`), including an explicit infeasible case (a shortfall no
  combination of capped levers can bridge) to confirm the solver's infeasible-status path is
  exercised, not just assumed to work.
