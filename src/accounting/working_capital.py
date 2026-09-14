"""Working capital metrics (§27), built on top of the Phase 4 Income
Statement and Balance Sheet -- no new data, just ratios of what those two
statements already computed.

    DSO = (Accounts Receivable / Revenue) x Days in month
    DPO = (Accounts Payable / (COGS + Operating Expense)) x Days in month
    DIO = (Inventory / COGS) x Days in month
    CCC = DSO + DIO - DPO

DPO deliberately departs from the textbook "AP / COGS" formula. In this
business, Accounts Payable funds a broad base of vendor spend -- inventory
purchases AND most operating expense categories (rent, software, marketing,
logistics, professional services, utilities; see transaction_events.py) --
not just COGS-related purchasing. Dividing AP by COGS alone (the textbook
formula, which assumes AP is dominated by inventory purchases) inflated DPO
to 150-250+ days here, 3-5x an already-generous expectation; dividing by
(COGS + Operating Expense) -- everything that actually flows through AP,
give or take Salaries, which is paid directly and never touches AP --
brings it back into a believable range. This is a standard variant used for
opex-heavy or services businesses, not an arbitrary adjustment; see
CLAUDE.md for the numbers that motivated it.

These are point-in-time (single-month) ratios; a trailing 3-month average
of each is also computed to smooth month-to-month noise for trend charts,
since a single month's Revenue/COGS can move a lot (see the deliberate
late-window slowdown in Phase 2).

Run: python -m src.accounting.working_capital
"""

from __future__ import annotations

import pandas as pd

from src.accounting.balance_sheet import build_balance_sheet
from src.accounting.income_statement import build_income_statement, _load_processed
from src.common.config import PROJECT_ROOT
from src.common.logging_config import get_logger

log = get_logger("working_capital")

OUTPUT_PATH = PROJECT_ROOT / "reports" / "outputs" / "working_capital.csv"
SMOOTHING_WINDOW = 3


def build_working_capital(income_statement: pd.DataFrame, balance_sheet: pd.DataFrame) -> pd.DataFrame:
    wc = income_statement[["entity_id", "period", "revenue", "cogs"]].merge(
        balance_sheet[["entity_id", "period", "accounts_receivable", "inventory", "accounts_payable"]],
        on=["entity_id", "period"], how="inner",
    ).sort_values(["entity_id", "period"]).reset_index(drop=True)

    wc = wc.merge(
        income_statement[["entity_id", "period", "operating_expense"]], on=["entity_id", "period"], how="left",
    )

    days = wc["period"].apply(lambda p: p.days_in_month)
    wc["dso"] = (wc["accounts_receivable"] / wc["revenue"]) * days
    wc["dpo"] = (wc["accounts_payable"] / (wc["cogs"] + wc["operating_expense"])) * days
    wc["dio"] = (wc["inventory"] / wc["cogs"]) * days
    wc["ccc"] = wc["dso"] + wc["dio"] - wc["dpo"]

    for metric in ["dso", "dpo", "dio", "ccc"]:
        wc[f"{metric}_3m_avg"] = wc.groupby("entity_id")[metric].transform(
            lambda s: s.rolling(SMOOTHING_WINDOW, min_periods=1).mean()
        )

    ordered = [
        "entity_id", "period", "revenue", "cogs", "operating_expense",
        "accounts_receivable", "inventory", "accounts_payable",
        "dso", "dpo", "dio", "ccc", "dso_3m_avg", "dpo_3m_avg", "dio_3m_avg", "ccc_3m_avg",
    ]
    return wc[ordered]


def working_capital_commentary(wc: pd.DataFrame) -> list[str]:
    """Data-driven sentences (§43 Business Interpretation) -- every
    statement below is computed from wc, nothing is a canned template
    filled with placeholder numbers."""
    notes = []
    for entity_id, group in wc.groupby("entity_id"):
        first, last = group.iloc[0], group.iloc[-1]
        dso_change = last["dso_3m_avg"] - first["dso_3m_avg"]
        ccc_change = last["ccc_3m_avg"] - first["ccc_3m_avg"]
        direction = "deteriorated" if dso_change > 0 else "improved"
        notes.append(
            f"{entity_id}: DSO {direction} from {first['dso_3m_avg']:.0f} to {last['dso_3m_avg']:.0f} days "
            f"({dso_change:+.0f}) over the observed period; Cash Conversion Cycle moved from "
            f"{first['ccc_3m_avg']:.0f} to {last['ccc_3m_avg']:.0f} days ({ccc_change:+.0f})."
        )

    worst_ccc = wc.loc[wc["ccc_3m_avg"].idxmax()]
    notes.append(
        f"Highest observed Cash Conversion Cycle: {worst_ccc['entity_id']} in {worst_ccc['period']} at "
        f"{worst_ccc['ccc_3m_avg']:.0f} days (3-month average) -- the longer this is, the more cash is "
        "tied up in the operating cycle before it converts back to cash."
    )
    return notes


def main() -> None:
    transactions, coa, fx_rates = _load_processed()
    income_statement = build_income_statement(transactions, coa, fx_rates)
    balance_sheet = build_balance_sheet(transactions, coa, fx_rates, income_statement)
    wc = build_working_capital(income_statement, balance_sheet)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    wc.to_csv(OUTPUT_PATH, index=False)
    log.info("Wrote working capital metrics (%d entity-months) to %s", len(wc), OUTPUT_PATH.relative_to(PROJECT_ROOT))

    for note in working_capital_commentary(wc):
        log.info(note)


if __name__ == "__main__":
    main()
