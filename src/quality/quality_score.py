"""Orchestrates the full quality engine: ingest raw -> run every check ->
split into VALIDATED (data/processed/) and QUARANTINED (data/quarantine/)
-> compute the 0-100 Financial Data Quality Score from the *actual*
validation results (§23 -- never hardcoded).

Run: python -m src.quality.quality_score
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from src.common.config import load_config, PROJECT_ROOT
from src.common.logging_config import get_logger
from src.data.ingestion import load_all_raw
from src.quality import schema, completeness, duplicates, validity, consistency, reconciliation

log = get_logger("quality_score")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
QUARANTINE_DIR = PROJECT_ROOT / "data" / "quarantine"
REPORT_PATH = PROJECT_ROOT / "reports" / "outputs" / "quality_report.json"

ROW_LEVEL_DATASETS = ["transactions", "accounts_receivable", "accounts_payable"]
QUARANTINE_SEVERITIES = {"CRITICAL", "HIGH"}
SEVERITY_WEIGHT = {"CRITICAL": 1.0, "HIGH": 0.6, "MEDIUM": 0.3, "LOW": 0.1, "INFO": 0.02}

DIMENSION_RULE_PREFIXES = {
    "completeness": ("missing_",),
    "uniqueness": ("duplicate_",),
    "validity": (
        "invalid_currency", "negative_amount_not_allowed", "invalid_date",
        "payment_before_invoice", "amount_mismatch", "statistical_outlier", "invalid_due_date",
    ),
    "consistency": ("invalid_account", "invalid_entity", "invalid_customer_id", "invalid_vendor_id",
                     "inconsistent_classification"),
    "reconciliation": ("journal_entry_unbalanced", "subledger_tie_out_failure"),
}


def _classify_dimension(rule: str) -> str | None:
    for dimension, prefixes in DIMENSION_RULE_PREFIXES.items():
        if any(rule == p or rule.startswith(p) for p in prefixes):
            return dimension
    return None


def run_all_checks(datasets: dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    frames = []
    for name in ROW_LEVEL_DATASETS:
        frames.append(schema.check_schema(datasets[name], name))
    frames.append(completeness.check_completeness(datasets, config))
    frames.append(duplicates.check_duplicates(datasets, config))
    frames.append(validity.check_validity(datasets, config))
    frames.append(consistency.check_consistency(datasets, config))
    frames.append(reconciliation.check_reconciliation(datasets, config))
    issues = pd.concat(frames, ignore_index=True)
    issues["dimension"] = issues["validation_rule"].map(_classify_dimension)
    return issues


def split_validated_and_quarantine(
    datasets: dict[str, pd.DataFrame], issues: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    row_issues = issues[issues["row_uid"].notna()]
    quarantine_uids = set(
        row_issues.loc[row_issues["severity"].isin(QUARANTINE_SEVERITIES), "row_uid"]
    )

    validated, quarantined = {}, {}
    for name in ROW_LEVEL_DATASETS:
        df = datasets[name]
        is_quarantined = df["row_uid"].isin(quarantine_uids)
        validated[name] = df.loc[~is_quarantined].reset_index(drop=True)
        quarantined[name] = df.loc[is_quarantined].reset_index(drop=True)

    for name, df in datasets.items():
        if name not in ROW_LEVEL_DATASETS:
            validated[name] = df.reset_index(drop=True)

    return validated, quarantined


def build_quarantine_ledger(quarantined: dict[str, pd.DataFrame], issues: pd.DataFrame) -> dict[str, pd.DataFrame]:
    timestamp = datetime.now(timezone.utc).isoformat()
    ledgers = {}
    for name, df in quarantined.items():
        if df.empty:
            ledgers[name] = pd.DataFrame(columns=["record_id", "validation_rule", "severity", "reason", "timestamp"])
            continue
        relevant = issues[
            (issues["dataset"] == name) & issues["row_uid"].isin(df["row_uid"])
            & issues["severity"].isin(QUARANTINE_SEVERITIES)
        ]
        ledger = relevant[["row_uid", "validation_rule", "severity", "reason"]].rename(
            columns={"row_uid": "record_id"}
        )
        ledger["timestamp"] = timestamp
        ledgers[name] = ledger.reset_index(drop=True)
    return ledgers


def compute_quality_score(issues: pd.DataFrame, datasets: dict[str, pd.DataFrame], config: dict) -> dict:
    total_rows = sum(len(datasets[name]) for name in ROW_LEVEL_DATASETS)
    weights = config["quality"]["score_weights"]

    dimension_scores = {}
    for dimension in weights:
        dim_issues = issues[issues["dimension"] == dimension]
        weighted_penalty = dim_issues["severity"].map(SEVERITY_WEIGHT).fillna(0.3).sum()
        penalty_rate = min(weighted_penalty / total_rows, 1.0) if total_rows else 0.0
        dimension_scores[dimension] = round(100 * (1 - penalty_rate), 2)

    overall = round(sum(dimension_scores[d] * w for d, w in weights.items()), 2)
    return {"overall_score": overall, "dimension_scores": dimension_scores}


def write_layers(validated: dict[str, pd.DataFrame], quarantine_ledgers: dict[str, pd.DataFrame]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in validated.items():
        df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        log.info("Wrote validated data/processed/%s.csv (%d rows)", name, len(df))
    for name, df in quarantine_ledgers.items():
        df.to_csv(QUARANTINE_DIR / f"{name}.csv", index=False)
        log.info("Wrote quarantine data/quarantine/%s.csv (%d rows)", name, len(df))


def run_quality_engine() -> dict:
    config = load_config()
    datasets = load_all_raw()

    log.info("Running quality checks across %d datasets", len(datasets))
    issues = run_all_checks(datasets, config)
    log.info("Found %d issues (%d row-level)", len(issues), issues["row_uid"].notna().sum())
    log.info("Issues by severity: %s", issues["severity"].value_counts().to_dict())

    validated, quarantined = split_validated_and_quarantine(datasets, issues)
    quarantine_ledgers = build_quarantine_ledger(quarantined, issues)
    total_quarantined = sum(len(df) for df in quarantined.values())
    log.info("Quarantined %d records across %s", total_quarantined, ROW_LEVEL_DATASETS)

    score = compute_quality_score(issues, datasets, config)
    log.info("Financial Data Quality Score: %.1f/100 -- %s", score["overall_score"], score["dimension_scores"])

    write_layers(validated, quarantine_ledgers)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_counts_raw": {name: len(datasets[name]) for name in ROW_LEVEL_DATASETS},
        "row_counts_validated": {name: len(validated[name]) for name in ROW_LEVEL_DATASETS},
        "row_counts_quarantined": {name: len(quarantined[name]) for name in ROW_LEVEL_DATASETS},
        "issue_counts_by_severity": issues["severity"].value_counts().to_dict(),
        "issue_counts_by_rule": issues["validation_rule"].value_counts().to_dict(),
        **score,
    }
    with REPORT_PATH.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Wrote quality report to %s", REPORT_PATH.relative_to(PROJECT_ROOT))

    return report


def main() -> None:
    run_quality_engine()


if __name__ == "__main__":
    main()
