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
- [ ] Phase 3 — data quality engine
- [ ] Phase 4 — financial statements
- [ ] Phase 5 — provisions / accruals
- [ ] Phase 6 — working capital
- [ ] Phase 7 — forecasting
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
