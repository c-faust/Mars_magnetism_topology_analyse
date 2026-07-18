# MAVEN region_id classifier

This module classifies each requested UTC sample into:

| region_id | Region |
|---:|---|
| 0 | Unknown / boundary / unresolved |
| 1 | Solar wind |
| 2 | Magnetosheath |
| 3 | Ionosphere |
| 4 | Magnetic lobes |

The runtime hierarchy uses the Gruesbeck et al. (2018) three-dimensional MSO
bow-shock surface by default, the Vignes et al. (2000) average MPB conic, MAVEN
MAG position and field data, and optional SWEA/STATIC support. Statistical
surfaces and empirical thresholds are classification guides rather than exact
instantaneous boundaries. Samples near either boundary are therefore assigned
to `Unknown`.

`region_id_implementation_status.csv` records which rule-table features are
currently active, diagnostic-only, or still pending calibration.

The complete runtime decision order, equations, and numeric thresholds are in
`REGION_ID_DECISION_FLOW.md`.
The disposition of the latest algorithm review is recorded in
`EVALUATION_RESPONSE.md`.

Run:

```powershell
python region_id\classify_region_id.py `
  --start 2016-10-06T18:02:00 `
  --end 2016-10-06T18:20:00 `
  --cadence-seconds 10
```

Outputs are written below `outputs/region_id/<start>_<end>/`:

- `region_id_timeseries.csv`: classifications and diagnostic features
- `region_id_timeseries.png`: time on x, `region_id` on y
- `region_id_summary.json`: models, thresholds, source files, region/reason
  counts, Unknown fraction, and geometry-only count

The source-to-rule mapping remains in `region_id_rules.csv`.
