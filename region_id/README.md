# MAVEN region_id classifier

This module classifies each requested UTC sample into:

| region_id | Region |
|---:|---|
| 0 | Unknown / conflicting / unresolved |
| 1 | Solar wind |
| 2 | Magnetosheath |
| 3 | Ionosphere |
| 4 | Magnetic lobes |

The runtime hierarchy uses the Gruesbeck et al. (2018) three-dimensional MSO
bow-shock surface, the Vignes et al. (2000) average MPB conic, MAG, SWIA survey
moments/spectra, SWEA and optional STATIC support. IDs 1-4 are proposed and
ranked entirely from in-situ plasma, particle, MAG and physical-position
evidence. The statistical bow-shock/MPB surfaces are only a small posterior
confidence term (default `+0.03`, hard-capped at `+0.05`): they cannot create a
candidate, reject a candidate, select between conflicting candidates, or force
a near-boundary sample to `Unknown`.

Magnetosheath recognition has two paths:

1. With a reliable local/bracketing upstream segment, require normalized
   density and magnetic compression plus bulk-flow deceleration.
2. If the orbit does not reach reliable upstream solar wind, require at least
   three local primary signatures among slow flow, proton heating, flow
   deflection, magnetic fluctuations and broad SWIA ion spectrum.

The upstream reference is therefore optional. Missing it does not invalidate a
whole orbit. STATIC H+ dominance can support ID 2, but missing STATIC is not a
hard failure. Current-sheet detections are written to `structure_flags` and do
not overwrite the background `region_id`. The CSV columns
`region_candidate_ids`, `region_candidate_scores`,
`boundary_geometry_support` and `boundary_geometry_confidence_bonus` make this
separation auditable per sample.

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
  --cadence-seconds 10 `
  --boundary-geometry-confidence-bonus 0.03 `
  --region-evidence-min-score-separation 0.08
```

Outputs are written below `outputs/region_id/<start>_<end>/`:

- `region_id_timeseries.csv`: classifications and diagnostic features
- `region_id_timeseries.png`: time on x, `region_id` on y
- `region_id_summary.json`: models, thresholds, source files, upstream segment
  provenance, region/reason counts and Unknown fraction

The source-to-rule mapping remains in `region_id_rules.csv`.
