# Region classifier evaluation response

This file records how the current implementation handles the algorithm reviews
supplied on 2026-07-18.

## Implemented now

| Review item | Action |
|---|---|
| Magnetosheath could be overwritten by ionosphere evidence | The decision order is now strictly nested: outside bow shock -> ID 1; inside bow shock but outside MPB -> ID 2; inner-region particle rules are evaluated only inside MPB. |
| Low altitude alone was too strong for ID 3 | `altitude <= 400 km` is now only a prior. ID 3 additionally requires cold planetary-heavy ions or photoelectrons. Low altitude alone returns ID 0. |
| Current sheet should precede ionosphere and lobe rules | An identified nightside current-sheet signature now returns ID 0 before either ID 3 or ID 4 can be assigned. |
| Current sheet depended on `B_now` hitting the field minimum | Detection now uses the angle between pre/post 20 s median vectors and the minimum field in a central +/-10 s window. |
| Stable field alone could overproduce ID 4 | ID 4 now also requires tail alignment >=0.50 and matched SWEA particle-exclusion evidence. |
| STATIC quality could not be diagnosed | CSV output now includes valid-bin counts, total valid ion flux, and integrated planetary-heavy-ion flux. |
| First-match returns were hard to audit | Summary JSON now includes counts by `reason`, Unknown fraction, and geometry-only count. |
| ID 0 might be hidden by plotting | Both plotting paths were verified to use discrete step lines and explicit ID 0 scatter points; zero is not masked or interpolated. |
| Geometry-only results should be explicit | Output CSV rows assigned ID 1 or ID 2 now set `geometry_only=true`. |
| Invalid MAG reasons were too broad | The output now distinguishes `missing_mag_sample`, `mag_time_mismatch`, `invalid_position`, and `invalid_magnetic_field`. |

## Deferred until supporting data or calibration are available

| Review item | Why it is not a hard rule yet |
|---|---|
| Minimum STATIC counts or flux for cold-heavy-ion evidence | The diagnostic values are now exported, but labeled-event calibration is still required before choosing hard minima. |
| Spacecraft-potential correction | No validated potential-correction path is connected to the STATIC feature extractor. |
| Solar-wind and magnetosheath plasma consistency | Suitable SWIA moments are not available in the current classifier inputs. |
| Lobe density and crustal-field exclusion | Tail alignment is now active, but density and crustal-field fraction require additional inputs and a crustal-field model. |
| Signed local-normal boundary distance | Boundary APIs currently expose radial offset. A local-normal distance implementation must be validated against both surface geometries. |
| Changing the fixed boundary buffer | The default remains 100 km until sensitivity tests at multiple margins establish a defensible value. |

The deferred items remain visible as scientific limitations rather than being
represented by unvalidated numerical thresholds.

## Geometry-order validation

The geometry-order revision was run over 2016-10-06 at 60 s cadence using the
local MAG, SWEA, and STATIC products. Before the later current-sheet and lobe
revision, it produced 1,440 rows with counts:

```text
ID 0: 115
ID 1: 228
ID 2: 625
ID 3: 177
ID 4: 295
```

The run confirmed these invariants:

- No sample outside the MPB was assigned ID 3.
- At 2016-10-06T13:24:00Z, cold-heavy-ion evidence was present outside the MPB;
  the geometry-first rule correctly retained ID 2.
- At 2016-10-06T04:35:00Z, altitude was 393.3 km without sufficient particle
  evidence; the sample correctly returned ID 0.
- Every ID 1 and ID 2 row set `geometry_only=true`, and no other region did.

The reproducible outputs are under
`outputs/region_id/evaluation_order_fix_20161006/`.

## Unknown-detection validation

The final revised classifier was rerun over the same day at 60 s cadence:

```text
ID 0: 210
ID 1: 228
ID 2: 625
ID 3: 177
ID 4: 200
```

Compared with the preceding version, 95 former ID 4 samples became ID 0:

```text
63 lacked matched particle-exclusion evidence
32 failed the tail-alignment or another stable-lobe-field condition
```

At 10 s cadence, the MAG-only daily scan found 13 current-sheet candidate
timestamps. In the 2016-10-06T18:49:00Z to 18:52:00Z event, the classifier
marked 18:50:30Z through 18:50:50Z as ID 0. Their old pointwise
`B_now/B_median` values were about 0.92, 0.64, and 1.02, while the new
pre/post rotations were 100-107 deg and central dip ratios were 0.40-0.60.
This directly reproduces the sampling-offset failure described in the review.

The ID 3 quality scan also confirmed why a hard STATIC threshold should not be
guessed: among 177 daily ID 3 rows, valid-bin count ranged from 8 to 794 and
total valid ion flux ranged from about `2.0e5` to `5.1e10`. These diagnostics
are now available for label-based calibration, but the single-day distribution
is not treated as an instrument-quality standard.

A geometry/plasma conflict scan found photoelectron flags in 11 solar-wind
geometry rows and 183 magnetosheath geometry rows, but no row combined that
flag with cold-heavy-ion evidence. A photoelectron flag alone is therefore not
used to force ID 1/2 to Unknown.

The outputs are under:

```text
outputs/region_id/evaluation_unknown_review_20161006/
outputs/region_id/current_sheet_validation_20161006_1850/
```

## Cadence and boundary sensitivity

For 2016-10-06T18:02:00Z to 18:20:00Z:

| Cadence | Boundary margin | Unknown rows | Unknown fraction |
|---:|---:|---:|---:|
| 2 s | 100 km | 126 / 541 | 23.29% |
| 10 s | 100 km | 26 / 109 | 23.85% |
| 10 s | 200 km | 32 / 109 | 29.36% |
| 10 s | 300 km | 37 / 109 | 33.94% |

The near-equal 2 s and 10 s fractions show that this particular interval is
dominated by sustained Unknown regions rather than sub-10-second gaps. The 2 s
run still provides more accurate transition times. Increasing the radial
buffer has a substantial and monotonic effect, so the default remains 100 km
until broader labeled-event sensitivity tests are available.

These outputs are under
`outputs/region_id/sensitivity_20161006_1802_1820/`.
