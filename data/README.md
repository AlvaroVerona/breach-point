# Data layers

Breach Point never silently deletes or overwrites bad data. Three layers, with
full lineage between them:

- **`raw/`** — exactly what the source systems produced, including
  intentionally injected data quality issues (duplicates, missing values,
  invalid values, broken journal entries). Never modified after generation.
- **`processed/`** — the *validated* subset: records that passed the quality
  engine's critical checks, ready to feed financial statements, forecasting,
  simulation and optimization.
- **`quarantine/`** — records that failed one or more critical validation
  rules, each tagged with `record_id`, `validation_rule`, `severity`,
  `reason`, `timestamp`. Nothing here is deleted; it is preserved so the
  failure can be traced back to its source record in `raw/`.

Regenerate everything with `make generate-data`; validate with `make
validate`. Files in this directory are gitignored (see `.gitignore`) — only
this README and the folder structure are committed.
