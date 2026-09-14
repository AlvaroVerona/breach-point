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
- **Inventory/COGS routing** (`src/data/transaction_events.py`): "Raw
  Materials" is NOT posted as a direct opex line. COGS targets post
  Dr COGS / Cr Inventory (consumption); a matching AP-funded purchase posts
  Dr Inventory / Cr Accounts Payable (replenishment, ~COGS ± noise). This
  keeps Inventory a real rolling balance instead of a frozen opening value,
  without double-counting the "Raw Materials" opex target (which is
  intentionally left unconsumed by the ledger).
- **Multi-leg entries balance to the cent via a "penny plug" on the last
  leg** (`LedgerBuilder.post_multi` in `src/data/ledger.py`) — rounding each
  leg independently first, then pushing the residual onto the last leg,
  sign-aware by which side (debit/credit) that leg is on. Get the sign
  wrong here and every multi-leg document is off by a cent; regression-
  tested in `test_small_ledger_documents_balance`.
- **Opening Retained Earnings is a legitimate one-time plug**
  (`opening_balances()` in `ledger.py`): computed once per entity as
  `assets - liabilities - common stock` to seed a balanced day-1 trial
  balance, representing accumulated history before the observed window.
  This is standard practice for seeding a new ledger, not a fabricated
  result — every period *after* day 1 reconciles through real double-entry
  postings, not through plugging.
- **row_uid lineage** (`src/data/ingestion.py`): assigned at ingestion from
  file row position (`f"{DATASET}_{position:08d}"`), not baked into raw
  CSVs. Raw CSVs are read with `dtype=str` and dates are kept as strings on
  ingestion — parsing dates immediately would turn an intentionally invalid
  string like "2024-02-30" into the same NaT as a genuinely missing date,
  destroying the distinction `validity.check_invalid_dates` needs to draw.
  All dates are written as `YYYY-MM-DD`, so quality checks parse with
  `format="%Y-%m-%d", errors="coerce"` rather than format-sniffing.
- **Quarantine threshold**: a record is quarantined if it has at least one
  CRITICAL or HIGH severity issue; MEDIUM/LOW/INFO issues stay in the
  validated layer (still logged, still count against the quality score).
  `data/quarantine/*.csv` is a ledger, not a 1:1 record dump — a record
  flagged for N reasons gets N rows sharing one `record_id`.
- **Quality score denominator**: every dimension's penalty is divided by
  `len(transactions) + len(accounts_receivable) + len(accounts_payable)`
  (one shared denominator across all 5 dimensions), not a dimension-
  specific count — simple and transparent over precise but harder to
  explain. See `compute_quality_score` in `src/quality/quality_score.py`.
- **Quarantine cascades at the document level for transactions**
  (`add_document_cascade_issues` in `src/quality/quality_score.py`): if any
  leg of a journal document is quarantined for *any* reason, every other
  leg of that same document is quarantined too (via a synthetic
  `document_quarantine_cascade` issue, so it still shows up in the
  quarantine ledger and the reconciliation score). Without this, row-level
  quarantine could orphan a balanced document's surviving leg, breaking
  Assets = Liabilities + Equity on the VALIDATED layer for a reason
  unrelated to any deliberately injected imbalance. Regression-tested in
  `test_validated_transactions_have_no_unbalanced_documents`.
- **Outlier / near-duplicate detection must group by a fine-grained key.**
  Grouping transactions by `account_category` (6 buckets) or comparing
  invoices across the full 3-year window produced 12k+/13k+ false
  positives purely from bucketing artifacts. Fixed: outliers group by
  `account_name` (transactions) / `expense_category` (AP) / `business_unit`
  (AR); near-duplicates require the *same calendar day* (not a sliding
  window) plus amount within 0.5%. If you touch these, re-check the ratio
  of flagged rows against total rows — it should stay in the low single
  digits, not 10%+.
- **Opening-balance documents are excluded from ALL injection**
  (`generate_data.py`: split out by `document_id.str.startswith("OPEN_")`
  before every `quality_injection.*` call on `transactions`, recombined
  after). Found by building Phase 4's Balance Sheet: one row's
  `transaction_id` got blanked by the generic missing-value injection,
  the document-cascade quarantine (correctly) removed the whole 13-line
  opening entry for that entity, and the entity's Balance Sheet ended up
  reconciling PERFECTLY but starting from zero instead of its true opening
  position (debt/common_stock stuck at 0 forever, since nothing else ever
  posts to those accounts). Regression-tested in
  `test_balance_sheet_opening_documents_survive_quarantine`.
- **Corporate income tax is not journaled; it's estimated at
  statement-build time** (`CORPORATE_TAX_RATE_BY_ENTITY` in
  `src/accounting/chart_of_accounts.py`) as a flat statutory rate on
  pre-tax income. This is the single most important Balance Sheet gotcha
  in this codebase: a notional, non-journaled deduction to Net Income
  lowers Retained Earnings with no offsetting entry anywhere, so the
  Balance Sheet drifts further out of balance every single month by
  exactly the cumulative tax amount (confirmed exactly: month-by-month
  diff matched cumsum(taxes) to the cent). Fixed by adding an explicit
  `income_tax_payable` liability (cumulative taxes) in `balance_sheet.py`
  -- standard accrual treatment, not a plug. Any other statement-level-only
  adjustment (not journaled in the ledger) needs the same treatment: a real
  offsetting Balance Sheet line, not just a P&L adjustment.
- **`economically_implausible` flag on the Balance Sheet**
  (`balance_sheet.py`): True when Cash/AR/Inventory/OtherCurrentAssets/
  FixedAssets goes negative, which can happen even on a perfectly
  reconciling Balance Sheet (Assets=Liabilities+Equity is an arithmetic
  identity; it says nothing about whether an individual account's sign
  makes economic sense). Currently flags 9/108 ENT_UK entity-months
  (negative Inventory) -- quarantine removes purchase-side records (more
  fields, more exposure to flags) faster than consumption-side ones for
  the entity with the least transaction volume; confirmed NOT a
  generation bug by checking raw pre-quarantine data, which shows healthy
  inventory throughout. Never silently floor a negative balance to 0 to
  hide this -- that breaks the equation without a matching adjustment,
  exactly like the tax bug above.

- **Accrual methodology is "expected vs. recognized," not a fabricated
  lag** (`src/provisions/accrual_model.py`). The generated ledger has no
  artificial invoice-arrival delay — every AP invoice is dated within the
  month it belongs to (Phase 2) — so there is no "missing invoice" gap to
  recover. Accruals here are legitimately `rolling_average(expected) -
  actual_recognized`, i.e. real month-to-month estimation variance in a
  recurring cost, matching the literal formula in the spec (§26) rather
  than inventing incompleteness that doesn't exist in the data. If you
  ever add a real invoice-arrival lag to Phase 2, this module's meaning
  changes materially — re-read this note before touching either.
- **Rolling average (window=3) is the primary `expected_cost`**; historical
  (expanding-window) and seasonal (same calendar month, prior years)
  averages are computed and reported alongside it but not used, so the
  choice of primary method stays auditable rather than hidden. Seasonal
  average is NaN for the first ~12 months of any series (no prior
  occurrence of that calendar month yet) — expected, not a bug.

- **AR write-off + tuned "stuck" payment rates** (`src/data/transaction_events.py`).
  Found while building Phase 6's DSO: no write-off mechanism meant ~7% of
  AR invoices were structurally stuck (governed by `COLLECTION_PROBABILITY`
  regardless of timing) with no resolution, so AR -- and DSO -- grew
  without bound over the 36-month window (985 invoices were already >6
  months overdue). Fixed with `BAD_DEBT_WRITEOFF_DAYS_PAST_DUE = 120`: an
  unpaid invoice past that threshold posts Dr Bad Debt Expense / Cr
  Accounts Receivable (a real ledger entry, `document_type="Bad Debt
  Writeoff"`, using the `Bad Debt Expense` account the chart of accounts
  reserved back in Phase 2 but never used until now). Also tuned
  `COLLECTION_PROBABILITY` 0.93->0.97 and added `AP_PAYMENT_PROBABILITY`
  =0.99 (was a bare 0.95 inline) -- AP has no write-off equivalent (a
  going concern eventually pays its vendors), so its stuck-rate has to
  stay much lower than AR's, or AP/DPO grows unboundedly the same way. If
  you ever touch these rates, regenerate the *entire* pipeline (data ->
  quality -> statements -> provisions -> working capital) and re-check
  DSO/DPO stay in a believable range (tens to ~150 days), not hundreds.
- **DPO uses `AP / (COGS + Operating Expense)`, not the textbook
  `AP / COGS`** (`src/accounting/working_capital.py`). This company's
  Accounts Payable funds a broad vendor base -- inventory purchases AND
  most opex categories (rent, software, marketing, logistics, professional
  services, utilities; only Salaries and Depreciation/Interest/Tax never
  touch AP) -- not just COGS-related purchasing the way a textbook
  retailer's AP would. A COGS-only denominator inflated DPO to 150-250+
  days here. If AP's composition changes (e.g. adding a new expense
  category that routes through AP), re-derive this denominator rather than
  assuming it still fits.

- **Forecast leakage prevention** (`src/forecasting/evaluation.py`):
  `evaluate_series` only ever passes `train` (the pre-validation prefix)
  into a model function -- never `val`. The validation-period forecast is
  therefore structurally incapable of seeing the values it's being scored
  against; regression-tested in
  `test_no_leakage_validation_forecast_depends_only_on_train` by corrupting
  the held-out values and confirming the forecast doesn't change. The
  *final* 12-month future forecast intentionally uses the full history
  (`series`, not `train`) -- that's correct, not leakage, since it's
  forecasting genuinely unseen future periods beyond the dataset.
- **Naive beats trend/seasonal models for Revenue on all three entities**
  -- not a bug. The 6-month validation window (`forecast.validation_months`
  in config) sits entirely inside Phase 2's deliberate late-window
  slowdown (`SLOWDOWN_START_MONTHS_FROM_END` in `financial_series.py`), so
  a model extrapolating the prior growth trend systematically overshoots
  right when the regime shifts, while "no change" happens to do less
  damage. If you change the slowdown's magnitude/timing, re-check whether
  model selection shifts back toward exponential smoothing/GBR -- that
  would also be a legitimate result, just a different one.
- **MAPE is unstable on Operating Cash Flow** (up to ~280% for one
  entity) because that series crosses near zero some months -- a
  division-by-near-zero artifact of the metric itself, not a forecast
  quality problem. Prefer RMSE or sMAPE (both also computed) when judging
  that series specifically.

- **`liquidity.minimum_cash` is €2,750,000, not the spec's illustrative
  €1,000,000 example** (`config/settings.yaml`). This company holds ~€18M
  cash against ~€1.37M/month consolidated operating outflow; €1M is under
  1 month of spend and would make every risk score trivially LOW regardless
  of scenario, defeating the point of simulating at all. Recalibrated to
  ~2 months of spend, a standard treasury buffer policy -- derived from
  actual data, not picked to hit a target score.
- **Liquidity risk comes out LOW in all three scenarios (0% breach
  probability), including pessimistic.** This is a genuine result, not a
  tuning failure -- the company is well-capitalized and profitable in this
  dataset, and worst simulated consolidated cash (~€13.3M under
  pessimistic) stays well above €2.75M. Do not "fix" this by inflating
  simulation volatility or further lowering minimum_cash to manufacture a
  breach; if Phase 9's optimization needs something to actively solve, use
  a clearly-labeled additional stress scenario instead, documented as a
  stress test, not the base case.
- **Monte Carlo simulation design** (`src/simulation/monte_carlo.py`):
  every stochastic input is empirically calibrated, not invented --
  Revenue/OpEx noise from each series' own Phase 7 validation residual
  std; DSO/DPO (customer/supplier payment timing) from their Phase 6
  historical distributions; COGS from simulated Revenue × historical
  margin distribution; AR/AP (and hence collections/payments) derived
  from simulated Revenue/COGS+OpEx and simulated DSO/DPO, so payment-
  timing risk has a direct, traceable cash effect rather than being folded
  into one generic noise term. Documented simplifications: draws are
  independent across metrics/entities/months (no shared macro shock/
  correlation structure), and Salaries (paid directly, no AP lag in the
  ledger) is folded into the same DPO-timed OpEx bucket as everything
  else. `run_simulation(context, scenario=...)` takes a precomputed
  `load_simulation_context()` so running base/optimistic/pessimistic
  doesn't refit Phase 7's GBR models three times.

- **Optimization uses `optimization.demo_minimum_cash` (€16M), not the
  real `liquidity.minimum_cash` (€2.75M)** (`src/optimization/
  cash_management.py`). The real policy has €11.3M of headroom even under
  the stress scenario -- confirmed by pushing revenue_pct down to -0.95
  and even -0.97 in ad-hoc testing, which still didn't breach the
  *consolidated* (cash-pooled across 3 entities) position, though each
  entity individually would have breached around -0.97 -- a genuine
  cash-pooling diversification benefit, not a bug. Rather than manufacture
  an artificial crisis with absurd stress assumptions, `run_optimization`
  reports `real_policy_headroom` (always positive, tested in
  `test_real_policy_headroom_is_positive`) and separately runs the demo
  optimization against the hypothetical stricter policy. Never conflate
  the two in reporting -- `demo_minimum_cash` is explicitly a "what if"
  exercise for the decision mechanism, not the actual liquidity policy.
- **`SAFETY_BUFFER_Z = 3.0`**: the LP's minimum-cash constraint is
  `demo_minimum_cash + SAFETY_BUFFER_Z * monthly_std`, where `monthly_std`
  is the Monte Carlo's own per-month cash standard deviation under the
  stress scenario. Without this buffer, the deterministic LP only
  protects the single expected path -- re-evaluating that "optimized"
  plan against actual Monte Carlo variance left breach probability at
  97.5% (barely improved from 100%). Z=1 and Z=2 were tried first (71%
  and 23% breach probability respectively) before settling on Z=3 (2.2%)
  -- if you change this, re-run and check the re-simulated breach
  probability actually drops to something you'd call "optimized," not
  just structurally different from before.
- **`compare_before_after` takes an already-run `mc_result`, not a
  scenario name** -- it must never re-run the simulation itself. The
  "before" figures need to be the exact same simulated paths the buffer
  (`monthly_std`) was computed from and that the "after" adjustment gets
  applied to; re-simulating with a fresh RNG draw would silently decouple
  before/after from a consistent set of paths. Regression-tested in
  `test_compare_before_after_uses_shared_simulation`.
- **The optimizer's plan is a static, pre-committed policy applied
  uniformly to every simulated path**, not a reactive one that adjusts to
  the realized cash trajectory -- a real treasury team observing actual
  results would adjust the plan monthly, which is a much harder
  (stochastic dynamic programming) problem, out of scope here. Documented
  in `compare_before_after`'s docstring, not hidden.

- **`app/` needs its own `__init__.py`** (`app/__init__.py`, empty) for
  `app/pages/*.py` to import `from app.components.data_loader import ...`
  -- without it, `streamlit run app/app.py` raises `ModuleNotFoundError:
  No module named 'app.components'; 'app' is not a package`. Found via
  browser testing before automating it in `tests/test_dashboard.py`'s
  `AppTest` runs; if you ever see that error again after restructuring
  `app/`, this is almost certainly why.
- **Dashboard reads `reports/outputs/` and `data/processed/`, never
  recomputes the pipeline live** (`app/components/data_loader.py`, all
  `st.cache_data`). Phase 7's forecasting alone fits a model per series;
  re-running that per page interaction would make the UI unusable. Run
  `make all` (or at least through the phase whose page you're viewing)
  before `make dashboard`.
- **Data Quality's quarantine table needs a join, not just the
  ledger** (`load_quarantine_detail` in `data_loader.py`): the quarantine
  ledger (`data/quarantine/*.csv`) only has `record_id, validation_rule,
  severity, reason, timestamp` -- no entity/source/account. Those come
  from `data/raw/*.csv`, joined on `record_id == row_uid` (row position,
  matching `src/data/ingestion.py`'s scheme). Recomputing row_uid here has
  to stay in sync with ingestion.py's `f"{name.upper()}_{i+1:08d}"` format.
- **Test dashboard pages with Streamlit's `AppTest`, not a live server.**
  `tests/test_dashboard.py` runs each page's actual script via
  `AppTest.from_file(...).run()` and asserts `not at.exception` --
  equivalent to a manual browser walkthrough but ~1s for all 7 pages
  instead of minutes, and it survives running under `pytest` without a
  browser or port. Use `at.dataframe`, `at.tabs`, etc. to assert on
  specific widgets when a test needs to check more than "didn't crash."

- **Test every `main()`, not just the functions it calls**
  (`tests/test_cli_entrypoints.py`). Every other test file calls a
  module's underlying functions directly (e.g. `build_balance_sheet(...)`),
  which never exercises `main()`'s own file-writing/orchestration code --
  that's the actual `make X` / `python -m src.X` path, and it's a
  different thing to get wrong (wrong output path, wrong dict keys handed
  to `json.dump`, etc.) than the logic underneath it. Coverage went from
  90% to 95% just from adding these. If you add a new `main()`, add its
  smoke test here too: call it, assert the promised output file(s) exist
  and are non-empty.
- **`test_no_pathological_accruals` compares against `expected_cost`
  (the rolling average), not `recognized_cost`** (`tests/
  test_provisions.py`). Comparing against recognized_cost directly failed
  on real data: a single month's recognized cost can legitimately dip
  close to zero, which makes even a normal-sized accrual look like a huge
  multiple of it. expected_cost is the stable scale of the series and
  doesn't have that problem.

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

## Phase 12 — final audit (done)

Verified in a **from-scratch clean clone** (fresh `git clone`, new venv,
`make install && make all`), not just the dev environment — which also
happened to pull newer major dependency versions (pandas 3.0.5 / numpy
2.5.3) than development used (2.x line). Every stage reproduced the exact
figures documented in README.md, 115/115 tests passed, 95% coverage, and
the dashboard started cleanly. README.md was rewritten from a phase-by-
phase build log into the polished, portfolio-facing document the spec
asks for (§52) — this file (CLAUDE.md) keeps the detailed, dated
engineering log; README.md keeps the results and methodology. If you
regenerate data/rerun the pipeline and get different headline numbers
than what's in README.md, that's a real signal something changed — check
whether it's an intentional code change (update the README) or a
regression (fix it), never just overwrite the README to match without
understanding why it moved.
