# region_id current decision flow and numeric criteria

This document describes the behavior currently implemented in
`classify_region_id.py` and `data_features.py`. It is an implementation
specification, not a list of independent suggestions from the literature.
The order below matters because the classifier returns as soon as one rule
matches.

## 1. Region IDs

| region_id | English name | Runtime meaning |
|---:|---|---|
| 0 | Unknown | Missing MAG position, boundary buffer, current sheet, or unresolved inner region |
| 1 | Solar wind | Outside the modeled bow shock |
| 2 | Magnetosheath | Inside the bow shock and outside the MPB |
| 3 | Ionosphere | Inside MPB with cold heavy ions, or low altitude plus photoelectrons |
| 4 | Magnetic lobes | Nightside, inside MPB, tail-aligned stable field, and particle-exclusion evidence |

## 2. Time sampling and data matching

For an interval `[start, end]`, classification targets are

```text
t_n = start + n * cadence_seconds
```

with `t_n <= end`. The default cadence is **10 s**. Cadence controls which
times are classified; it is not a 10 s average.

At each target time, the nearest instrument sample is used only if:

| Product | Maximum absolute time difference |
|---|---:|
| MAG SS 1 s | 2 s |
| SWEA `svypad` | 6 s |
| STATIC `c6-32e64m` | 12 s |

Input loading is padded beyond the requested interval by these matching
tolerances. MAG is padded by 30 s so the first and last target timestamps can
still use complete lobe and current-sheet windows. Only target times within the
requested start/end interval are written.

If no valid MAG position/field is available within 2 s, the output is
immediately `region_id=0`. Missing SWEA or STATIC does not by itself force
`region_id=0` for geometry-only ID 1 or ID 2. ID 4 additionally requires
matched SWEA evidence that excludes a photoelectron peak or identifies an
electron void.

## 3. Coordinates and basic geometry

All boundary models and spacecraft positions use MSO coordinates. The Mars
radius used by the code is

```text
R_M = 3389.5 km
```

For spacecraft position `(x, y, z)` in km:

```text
r_km       = sqrt(x^2 + y^2 + z^2)
altitude   = r_km - 3389.5 km
SZA        = acos(x / r_km), in degrees
(x,y,z)_RM = (x,y,z)_km / 3389.5
```

`SZA >= 100 deg` is the runtime nightside/tail condition for lobe and current
sheet tests. It is stricter than the geometric terminator at 90 degrees.

## 4. Boundary models

### 4.1 Bow shock: Gruesbeck et al. (2018), all-points MSO

The default three-dimensional statistical bow-shock surface is

```text
0.049*x^2 + 0.157*y^2 + 0.153*z^2
+ 0.026*x*y + 0.012*y*z + 0.051*x*z
+ 0.566*x - 0.031*y + 0.019*z = 1
```

where `x`, `y`, and `z` are in `R_M`. Define the left side minus 1 as
`F_bow(x,y,z)`:

```text
F_bow < 0  -> inside bow shock
F_bow > 0  -> outside bow shock
```

### 4.2 MPB: Vignes et al. (2000) average conic

The implemented axisymmetric MSO MPB uses

```text
x0      = 0.78 R_M
epsilon = 0.90
L       = 0.96 R_M
```

and the implicit equation

```text
sqrt((x - 0.78)^2 + y^2 + z^2) + 0.90*(x - 0.78) - 0.96 = 0
```

with coordinates in `R_M`:

```text
F_mpb < 0  -> inside MPB
F_mpb > 0  -> outside MPB
```

### 4.3 Boundary buffer

For both surfaces, the code intersects the Mars-center-to-spacecraft ray with
the model surface and calculates

```text
radial_offset_km = spacecraft_radius_km - boundary_radius_km
```

This is a radial offset along that ray, not the shortest three-dimensional
distance to the surface.

The default boundary buffer is:

```text
abs(radial_offset_km) <= 100 km
```

A sample in the bow-shock or MPB buffer is assigned `region_id=0` before the
plasma-region rules are evaluated.

## 5. MAG-derived numeric features

MAG features use a **60 s window**, centered on the target time
(`target +/- 30 s`).

### 5.1 Field magnitude and variability

```text
B_now             = norm(B at nearest target sample)
B_median          = median(norm(B) in 60 s window)
B_relative_std    = std(norm(B) in window) / B_median
```

At least three finite MAG vectors and `B_median > 0` are required for
`B_relative_std`.

### 5.2 Directional dispersion

Each finite, nonzero field vector is normalized to a unit vector. The mean unit
direction is calculated, followed by the angular separation `theta_i` of each
sample from that mean. The implemented dispersion is

```text
B_direction_dispersion_deg = sqrt(mean(theta_i^2))
```

### 5.3 Component reversal

Within the 60 s window:

1. Calculate `max(Bj)-min(Bj)` for `Bx`, `By`, and `Bz`.
2. Select the component with the largest range.
3. A reversal is present when `min(Bj) < 0 < max(Bj)`.

### 5.4 Current-sheet signature

For each target time, the MAG context is divided into:

```text
pre flank:     target - 30 s <= t < target - 10 s
center:        target - 10 s <= t <= target + 10 s
post flank:    target + 10 s < t <= target + 30 s
```

The median magnetic vectors in the pre and post flanks are `B_pre` and
`B_post`. The diagnostics are:

```text
rotation_angle =
    acos(dot(B_pre, B_post) / (norm(B_pre) * norm(B_post)))

center_min_B = min(norm(B) in center)
flank_median_B = median(norm(B) in both flanks)
dip_ratio = center_min_B / flank_median_B
```

The current-sheet flag is true only when:

```text
SZA >= 100 deg
rotation_angle >= 90 deg
dip_ratio <= 0.70
```

After the MPB geometry is resolved, the current-sheet rule is tested before
ionosphere and lobe rules. The centered 20 s interval makes the flag insensitive
to whether the classification timestamp lands exactly on the minimum field.
The older `component_reversal` and `B_now/B_median` values remain CSV
diagnostics but are no longer hard current-sheet conditions.

## 6. SWEA-derived numeric features

SWEA PAD flux is averaged over valid pitch-angle bins. A flux value is valid
when

```text
0 < flux < 1e30
```

### 6.1 Photoelectron peak

For each omnidirectional spectrum:

```text
C = median flux from 20 to 30 eV
L = median flux from 12 to 18 eV
R = median flux from 32 to 50 eV
S = max(valid L, valid R)
photoelectron_ratio = C / S
```

The flag is

```text
photoelectron_present = photoelectron_ratio >= 1.20
```

This flag is one of the two particle paths for low-altitude ID 3. An explicitly
absent peak in matched SWEA data can also provide the particle-exclusion
evidence required by ID 4.

### 6.2 Electron void

The SWEA energy channel nearest **40 eV** is selected:

```text
electron_void = 0 <= flux_near_40eV < 1.0e5
```

`electron_void=true` supplies particle-exclusion evidence for ID 4 and
increases its confidence.

### 6.3 Diagnostic-only suppression ratio

The code also records

```text
high_energy_suppression_ratio =
median_flux(20-30 eV) / median_flux(80-120 eV)
```

but this ratio does not currently change `region_id`.

## 7. STATIC-derived numeric features

A STATIC bin is valid when:

```text
0 < eflux < 1e30
energy > 0 eV
mass > 0 amu
```

Runtime population masks are:

```text
H+ population:                 0.5 <= mass <= 2.0 amu
planetary-heavy-ion population: mass >= 12.0 amu
```

The planetary-heavy-ion fraction is

```text
sum(valid eflux in mass >= 12 amu) / sum(all valid eflux)
```

The heavy-ion peak energy is the energy channel with the largest
mass-integrated planetary-heavy-ion flux. The cold-heavy-ion flag requires:

```text
planetary_heavy_ion_flux_fraction >= 0.45
heavy_ion_peak_energy_eV <= 10.0 eV
```

Both conditions must be finite and true.

STATIC also records:

```text
static_valid_bin_count
planetary_heavy_ion_valid_bin_count
total_valid_ion_flux
planetary_heavy_ion_integrated_flux
H+ fraction
H+ peak energy
H+ log-energy width
```

These quantities are diagnostic-only until minimum-statistics thresholds are
calibrated.

## 8. Exact decision order

The following pseudocode mirrors `classify_region_sample()`:

```text
1. if MAG position/field is invalid or farther than 2 s:
       return 0

2. if abs(bow radial offset) <= 100 km:
       return 0

3. if outside Gruesbeck 2018 bow shock:
       return 1

4. if abs(MPB radial offset) <= 100 km:
       return 0

5. if outside MPB:
       return 2

6. # From this point onward, the sample is inside MPB.

7. if current-sheet signature is true:
       return 0

8. low_altitude =
       finite(altitude) and altitude <= 400 km

9. cold_heavy_ions =
       heavy-ion fraction >= 0.45
       and heavy-ion peak energy <= 10 eV

10. if low_altitude and cold_heavy_ions:
       return 3

11. if cold_heavy_ions:
       return 3

12. if low_altitude and photoelectron_present:
        return 3

13. if low_altitude without either particle indicator:
        return 0

14. lobe_geometry =
        SZA >= 100 deg
        and altitude >= 400 km

15. stable_lobe_field =
        5 nT <= B_median <= 25 nT
        and B_relative_std <= 0.25
        and B_direction_dispersion <= 25 deg
        and abs(median Bx) / norm(median B) >= 0.50

16. lobe_particle_exclusion =
        electron_void is true
        or matched SWEA explicitly shows photoelectron_present == false

17. if lobe_geometry and stable_lobe_field and lobe_particle_exclusion:
        return 4

18. if lobe geometry is true but either support group fails:
        return 0

19. otherwise:
        return 0
```

Consequences of this priority:

- The bow-shock buffer is checked before Solar wind. After the sample is known
  to be inside the bow shock, the MPB buffer overrides all downstream classes.
- Bow-shock-inside plus MPB-outside geometry is locked to Magnetosheath before
  any ionosphere or lobe feature is considered.
- Inside MPB, a current-sheet signature overrides ionosphere and lobe features.
- Altitude at or below 400 km is a prior, not a sufficient ionosphere rule.
- A sample inside MPB is not automatically a lobe; an unresolved inner-region
  sample remains `region_id=0`.

## 9. Per-region criteria

### region_id 0: Unknown

Any one of the following runtime paths produces ID 0:

| Condition | Confidence | Reason string |
|---|---:|---|
| No MAG sample is available | 0.00 | `missing_mag_sample` |
| Nearest MAG time difference >2 s | 0.00 | `mag_time_mismatch` |
| Nonfinite/zero spacecraft position | 0.00 | `invalid_position` |
| Nonfinite magnetic field | 0.00 | `invalid_magnetic_field` |
| Bow radial distance within +/-100 km | 0.35 | `near_bow_shock` |
| MPB radial distance within +/-100 km | 0.35 | `near_magnetic_pileup_boundary` |
| Current-sheet signature | 0.40 | `nightside_current_sheet_signature` |
| Altitude <=400 km but neither cold heavy ions nor photoelectrons are present | 0.30 | `low_altitude_without_ionospheric_particle_evidence` |
| Stable lobe geometry/field but no matched particle-exclusion evidence | 0.30 | `stable_lobe_field_without_particle_exclusion` |
| Lobe geometry true but stable-field criteria fail | 0.30 | `nightside_inner_region_without_stable_lobe_field` |
| Inside MPB but no implemented inner-region rule matches | 0.25 | `inside_mpb_not_resolved_by_available_features` |

### region_id 1: Solar wind

Required runtime path:

```text
not within the +/-100 km bow-shock buffer
and bow_location == outside
```

The returned confidence is **0.98** and the reason is
`outside_gruesbeck_bow_shock`. No H+ beam, solar-wind dynamic pressure, EUV, or
magnetosonic Mach-number threshold is currently required.

The CSV field `geometry_only` is `true` for this result.

### region_id 2: Magnetosheath

Required runtime path:

```text
inside bow shock
not within either 100 km boundary buffer
MPB location == outside
```

The returned confidence is **0.86** and the reason is
`inside_bow_shock_outside_mpb`.

H+ dominance, magnetic compression, and magnetic turbulence are recorded or
planned diagnostics but are not hard ID 2 requirements in the current code.
The CSV field `geometry_only` is `true`.

### region_id 3: Ionosphere

ID 3 is evaluated only after the sample is inside MPB and current sheet has
been rejected.

#### Branch A: cold planetary-heavy ions

```text
planetary-heavy-ion fraction >= 0.45
and heavy-ion peak energy <= 10 eV
```

This branch can classify ID 3 above 400 km. Its confidence is **0.82** when
cold heavy ions are the only support.

When altitude is also `<=400 km`, confidence is **0.94**. If photoelectrons are
also present, confidence is **0.98**.

#### Branch B: low altitude plus photoelectrons

```text
altitude <= 400 km
and photoelectron_ratio >= 1.20
```

This branch has confidence **0.94**. There is no lower-altitude cutoff in the
runtime code.

`altitude <=400 km` without either cold heavy ions or photoelectrons now returns
ID 0 rather than ID 3. This prevents altitude alone from acting as a sufficient
ionosphere condition.

### region_id 4: Magnetic lobes

All geometry and field conditions are required:

```text
MPB location == inside
SZA >= 100 deg
altitude >= 400 km
5 nT <= 60-s median |B| <= 25 nT
std(|B|) / median(|B|) <= 0.25
RMS magnetic-direction dispersion <= 25 deg
abs(median Bx) / norm(median B) >= 0.50
current-sheet signature == false
electron_void == true
or matched SWEA explicitly shows photoelectron_present == false
```

The field-only state no longer produces ID 4. With required particle exclusion,
the confidence is:

```text
+0.05 if electron_void is true
+0.05 if matched SWEA data explicitly show photoelectron_present == false
base = 0.78
maximum confidence = 0.95
```

The practical current values are 0.83 or 0.88. Missing SWEA does not count as
explicit photoelectron absence and therefore leaves the sample at ID 0.

## 10. Default numeric configuration

| Configuration key | Default |
|---|---:|
| `cadence_seconds` | 10 s |
| `max_mag_delta_seconds` | 2 s |
| `max_swe_delta_seconds` | 6 s |
| `max_static_delta_seconds` | 12 s |
| `boundary_margin_km` | 100 km |
| `ionosphere_max_altitude_km` | 400 km |
| `heavy_ion_fraction_threshold` | 0.45 |
| `cold_heavy_ion_max_energy_eV` | 10 eV |
| `lobe_min_sza_deg` | 100 deg |
| `lobe_min_altitude_km` | 400 km |
| `lobe_min_b_nT` | 5 nT |
| `lobe_max_b_nT` | 25 nT |
| `lobe_max_b_relative_std` | 0.25 |
| `lobe_max_direction_dispersion_deg` | 25 deg |
| `lobe_min_tail_alignment` | 0.50 |
| `magnetic_window_seconds` | 60 s |
| `current_sheet_flank_window_seconds` | 20 s |
| `current_sheet_center_half_window_seconds` | 10 s |
| `current_sheet_min_rotation_deg` | 90 deg |
| `current_sheet_b_dip_ratio` | 0.70 |
| `photoelectron_ratio_threshold` | 1.20 |
| `electron_void_target_energy_eV` | 40 eV |
| `electron_void_flux_threshold` | 1.0e5 |

## 11. Scientific limitations of the current implementation

- The bow shock and MPB are statistical average surfaces, not instantaneous
  boundaries.
- Boundary buffering currently uses radial offset from Mars, not signed shortest
  distance along the local boundary normal. The fixed 100 km margin should be
  sensitivity-tested, for example at 100, 200, and 300 km, before scientific
  event statistics are reported.
- The Gruesbeck 2018 surface is not adjusted at runtime using EUV,
  magnetosonic Mach number, or solar-wind dynamic pressure.
- ID 1 and ID 2 are currently geometry-only classifications. Solar-wind and
  magnetosheath plasma consistency checks cannot be made until suitable SWIA
  moments or equivalent upstream measurements are connected.
- SWIA moments are not present in the current local classification inputs.
- STATIC cold-heavy-ion evidence currently has no minimum valid-bin count,
  minimum integrated flux, or spacecraft-potential correction. These require
  instrument-quality information and event calibration before hard thresholds
  are added.
- The lobe rule now requires tail alignment, but it does not yet exclude strong
  crustal-field regions or test plasma density. These require a crustal-field
  model and additional calibrated plasma inputs.
- The 0.45 heavy-ion fraction, lobe field range, variability, direction
  dispersion, and current-sheet dip ratio still require calibration against
  labeled events.
- `confidence` is a deterministic rule score, not a calibrated probability.
- Region transitions inside the 100 km buffers are intentionally reported as
  Unknown rather than forced into adjacent classes.
