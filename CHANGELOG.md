# Changelog

## 0.1.0 (2026-11-14)

- Extracted from the rcad kernel repository into a standalone library
  (same shape as `rcad-gcs`): own repository, own license, `[workspace]`
  root, consumed by rcad as a git dependency.
- M0 data model: `Body` / `Marker` / `Joint` / `Driver` / `ForceElement` /
  `System` with deterministic `BTreeMap` tables and `{code,detail,hint}`
  validation gates; serde golden round-trip tests.
- Dependency on `rcad-gcs` declared for the M1 position-level solver.
