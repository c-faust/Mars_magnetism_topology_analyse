# region_id current decision flow and numeric criteria

This document describes the rules implemented by `classify_region_id.py` and
`data_features.py`. The classifier identifies background regions only. A
current-sheet signature is retained in `structure_flags` and never replaces
the background `region_id`.

## Region IDs and data

| ID | Meaning | Main evidence |
|---:|---|---|
| 0 | Unknown / conflicting / unresolved | missing or contradictory evidence |
| 1 | Solar wind | normalized or reference-free solar-wind plasma |
| 2 | Magnetosheath | normalized or reference-free sheath plasma |
| 3 | Ionosphere | cold planetary heavy ions and/or low-altitude photoelectrons |
| 4 | Magnetic lobes | nightside stable field plus multichannel electron depletion |

The default target cadence is 10 s. Maximum nearest-sample differences are
2 s for MAG, 6 s for SWEA/SWIA and 12 s for STATIC. SWIA moment features are
robust medians in a centered 60 s window. A SWIA window is valid only when:

```text
nearest quality_flag == 1
nearest decom_flag == 1
atten_state != 3
valid sample count >= 5
valid coverage >= 0.60
maximum valid-sample gap <= 12 s
```

The onboard SWIA moments assume protons. Density and bulk velocity are primary
features. Temperature is auxiliary because alpha particles and field-of-view
coverage can bias the onboard temperature.

## Statistical boundaries are minor confidence support

Positions and boundaries are in MSO. The default bow shock is the Gruesbeck et
al. (2018) all-points surface and the MPB is the Vignes et al. (2000) average
conic. The Mars radius is 3389.5 km.

The bow-shock and MPB surfaces do not participate in candidate creation or
candidate ranking. First, IDs 1-4 are independently proposed from in-situ
observations. If the two highest observational scores differ by less than
`0.08`, the result is ID 0 due to conflicting region evidence; model geometry
is not allowed to break that conflict.

Only after an observational winner has been selected can clearly consistent
statistical geometry add the default `+0.03` confidence bonus:

```text
outside bow shock                         -> supports an existing ID 1
inside bow shock and outside MPB          -> supports an existing ID 2
inside MPB                                -> supports an existing ID 3 or ID 4
```

The configured bonus is hard-capped at `0.05`. A disagreement supplies no
bonus and no penalty. The default 100 km boundary margin means only that a
surface is too close to supply this bonus; it no longer forces ID 0. Geometry
alone never produces any of IDs 1-4 and never rejects an observational label.

## Optional upstream reference

The algorithm does not assume that every orbit reaches upstream solar wind.
High-confidence upstream samples must satisfy all of:

```text
valid SWIA proton moments
r >= 2.5 R_M
x_MSO >= 1.0 R_M
bulk speed >= 300 km/s
```

The modeled bow-shock location is recorded as the segment diagnostic
`bow_model_outside_fraction`, but it is not a segment-admission condition and
therefore cannot indirectly enable or disable normalized ID 1/2 evidence.

Samples are grouped with gaps no larger than 12 s. A segment requires at least
15 samples. For density, speed and |B|, its largest robust relative spread must
be no more than 0.35, where each spread is

```text
(P90 - P10) / (2 * median)
```

For each target, the source is selected in this order:

1. `local_upstream_segment` if the target is inside a segment;
2. `bracketing_upstream_segments` if both are within 6 h and mutually
   consistent to 0.35;
3. `nearest_upstream_segment` if one is within 6 h;
4. `unavailable` otherwise.

Inconsistent bracketing segments are explicitly rejected rather than averaged.
All source, age, spread, baseline and ratio fields are written to CSV.

## ID 1: solar wind

ID 1 requires either a normalized or local plasma signature, independent of
the statistical bow-shock location.

Normalized signature:

```text
0.70 <= n / n_up <= 1.30
0.70 <= B / B_up <= 1.30
0.85 <= V / V_up <= 1.15
```

Reference-free local signature:

```text
V >= 200 km/s
v_th,p / V <= 0.17
std(|B|) / median(|B|) <= 0.15
```

with

```text
v_th,p / V = 13.841 * sqrt(T_p[eV]) / V[km/s]
```

Bow-shock agreement only adds the small posterior confidence bonus described
above.

## ID 2: magnetosheath

ID 2 is evaluated from plasma evidence at every valid sample. Being between
the statistical bow-shock and MPB surfaces is optional confidence support, not
a prerequisite.

### Path A: upstream-normalized

All three conditions are required:

```text
n / n_up >= 1.50
B / B_up >= 2.00
0.35 <= V / V_up <= 0.85
```

Confidence is 0.90 for a local upstream reference and 0.84 for a borrowed
nearest/bracketing reference. Optional H+ support adds 0.02.

### Path B: reference-independent local evidence

This path does not consume an upstream baseline. It remains available when no
upstream segment exists and as a conservative fallback when a reference exists
but the three normalized conditions do not all pass.

At least three of these five primary local signatures are required:

| Signature | Default test |
|---|---|
| slow flow | `V <= 300 km/s` |
| proton heating | `v_th,p / V >= 0.22` |
| flow deflection | angle from nominal `-X_MSO >= 20 deg` |
| magnetic fluctuations | `std(|B|)/median(|B|) >= 0.15` |
| broad ion spectrum | SWIA log-energy width `>= 0.28` |

The spectral width is the differential-energy-flux-weighted standard deviation
of `log10(energy/eV)` over valid channels with positive counts. STATIC H+ flux
fraction `>=0.50` is recorded as optional support and does not count among the
three primary signatures.

If matched STATIC data instead have planetary-heavy-ion fraction `>=0.45` and
H+ fraction `<0.50`, they contradict this reference-free sheath path. Missing
STATIC never blocks ID 2. Confidence is 0.76 for three primary signatures and
0.82 for four or five.

## ID 3: ionosphere

ID 3 rules are evaluated independent of the statistical MPB location:

```text
cold planetary heavy ions:
    heavy-ion flux fraction >= 0.45
    heavy-ion peak energy <= 10 eV

or

low-altitude photoelectrons:
    altitude <= 400 km
    SWEA photoelectron ratio >= 1.20
```

Altitude alone is not sufficient.

## ID 4: magnetic lobes

The physical position, field and particle conditions are:

```text
SZA >= 100 deg
altitude >= 400 km
5 <= median(|B|) <= 25 nT
std(|B|)/median(|B|) <= 0.25
magnetic direction dispersion <= 25 deg
abs(median Bx)/norm(median B) >= 0.50
```

Particle exclusion now requires multichannel SWEA depletion, not absence of a
photoelectron peak and not a single 40 eV channel:

```text
30-80 eV valid channels >= 3
fraction below 1e5 >= 0.75
median flux below 1e5
```

If valid SWIA density exists it must also be `<=0.50 cm^-3`. Missing SWIA
density lowers confidence to 0.82 but does not masquerade as a measured void;
matched low density gives confidence 0.88.

The SZA and altitude criteria are retained because they describe the measured
spacecraft position and physical lobe context. The statistical MPB surface is
not required; MPB agreement only adds the small posterior confidence bonus.

## Current-sheet structure flag

The centered MAG windows remain:

```text
pre flank:  target-30 s to target-10 s
center:     target-10 s to target+10 s
post flank: target+10 s to target+30 s
```

The flag requires nightside SZA `>=100 deg`, pre/post median-vector rotation
`>=90 deg` and center minimum |B| divided by flank median |B| `<=0.70`.
Detection writes `current_sheet` to `structure_flags`; it does not force ID 0.

## Candidate conflict and main Unknown paths

Candidate observational scores are written before any boundary bonus. If the
two highest scores are separated by less than `0.08`, the sample becomes ID 0
with `conflicting_region_evidence`; bow shock and MPB cannot choose the winner.
ID 0 is also retained for missing MAG, insufficient/contradictory plasma or
particle evidence, altitude-only ionosphere contexts and unresolved samples.
`confidence` is a deterministic rule score, not a calibrated probability.

## Scientific limitations

- Bow-shock and MPB surfaces are statistical averages, not instantaneous
  boundaries.
- Upstream thresholds and local sheath thresholds need calibration on a
  manually labeled multi-season event set before publication use.
- SWIA temperature remains auxiliary; the onboard proton assumption and field
  of view must be considered.
- STATIC fractions are ratios of summed valid energy-flux array values, not a
  full species-distribution inversion, so they are support/contradiction only.
- No temporal label smoothing, boundary-crossing finder, MVA, or machine
  learning is applied in this revision.
