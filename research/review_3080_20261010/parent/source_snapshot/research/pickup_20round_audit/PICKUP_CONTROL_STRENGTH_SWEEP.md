# Pickup masked-reconstruction arm: control-strength mask-fraction sweep

Status: **exploratory dev pilot, not confirmatory**. Every interval below is DESCRIPTIVE
ONLY and carries no alpha. Report generated from
`control_strength_sweep_summary.json` by `summarize_control_strength_sweep.py` +
`render_control_strength_sweep_md.py`; no number here is hand-entered.

## Why this sweep exists

In both earlier pilots the distance-only `recon_only` control was nearly INERT: it
reproduced the baseline template ordering in 9 of 12 rows (|S|=3, `pickup_arm_pilot_v1`)
and 23 of 28 rows (|S|=4..5, `pickup_arm_pilot_n45_v1`), both at the registered
`mask_fraction=0.5`. A control that cannot change anything cannot demonstrate a difference
either way, so the existing `maskco - recon_only` contrasts were not interpretable as
evidence about the guide. Here the SAME operator is made freer by increasing the mask
fraction, which is the cheapest honest lever, and the SAME pilot is re-run over the SAME
states and seeds so the fractions are paired.

Definition used throughout: a control is **INERT** when its identity rate (installed
sequence == baseline template) is **above about 0.5**, i.e. it reproduces the baseline
ordering in the majority of rows.

## Registered sweep

| item | value |
| --- | --- |
| operator | `pickup_maskco_arm.py::propose`, UNCHANGED (SHA-256 2865CC17E2653DF2C113A03F046C5938F3DF376E2AE4F52B990C85A058A65D45) |
| harness | `pickup_arm_pilot.py`, UNCHANGED (SHA-256 28D3F5A5536B6D9946BCDF39AF35B4E11CACECCE92F2FB5BE9A892D3132A9CE8) |
| arm modes | `maskco` (guide used), `recon_only` (same mask schedule, same ONE decoder call, logits discarded), `discard` (identical to recon_only including paying the call) |
| states / seeds | 4 instances x up to 2 states, `--min-orders 3 --max-orders 5`, `--pilot-seed 2026120202 --seeds-per-mode 4` |
| seeds per fraction | `mask_fraction` in {0.50, 0.70, 0.80, 0.90, 1.00}; identical states, seeds, guide weight and one decoder call per proposal at every fraction |
| utility | full A-v1 utility of the whole day after installing the proposed permutation into the env's own recorded suffix template via `force_suffix` |
| bootstrap | 2000 percentile draws, seed 2026120212 (descriptive only) |

## Main table (one row per mask fraction)

| mask fraction | recon_only identity rate | discard identity rate | maskco identity rate | recon_only lead-order change rate | recon_only effective-perturbation rate | mean raw delta: maskco | mean raw delta: recon_only | mean raw delta: discard | maskco - recon_only (95% descriptive bootstrap CI) | pairs | states where the insertability guard fired (rows / states) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.50 | 0.8125 (26/32) | 0.8125 (26/32) | 0.0938 (3/32) | 0.1250 | 0.1250 | -5.16080 | 0.00222 | 0.00222 | -9.39378 [-15.96824, -2.83070] | 16 | 4 / 4 |
| 0.70 | 0.3750 (12/32) | 0.3750 (12/32) | 0.0000 (0/32) | 0.5000 | 0.5000 | -8.90508 | 0.00888 | 0.00888 | -14.09658 [-19.72060, -8.48096] | 16 | 0 / 0 |
| 0.80 | 0.1250 (4/32) | 0.1250 (4/32) | 0.0000 (0/32) | 0.8750 | 0.8750 | -5.62654 | -3.73916 | -3.73916 | -0.04460 [-9.43291, 9.33387] | 16 | 0 / 0 |
| 0.90 | 0.0000 (0/32) | 0.0000 (0/32) | 0.0000 (0/32) | 1.0000 | 1.0000 | -5.62144 | -7.49312 | -7.49312 | 7.47352 [3.72251, 11.22944] | 16 | 0 / 0 |
| 1.00 | 0.0000 (0/32) | 0.0000 (0/32) | 0.0000 (0/32) | 1.0000 | 1.0000 | -5.62144 | -7.49312 | -7.49312 | 7.47352 [3.72251, 11.22944] | 16 | 0 / 0 |

Counts behind the per-mode means (raw delta utility sign split), per fraction:

| mask fraction | mode | mean | positive / zero / negative | hard-feasible | failures | decoder calls |
| --- | --- | --- | --- | --- | --- | --- |
| 0.50 | maskco | -5.16080 | 7 / 16 / 9 | 32 | 0 | 32 |
| 0.50 | recon_only | 0.00222 | 3 / 28 / 1 | 32 | 0 | 32 |
| 0.50 | discard | 0.00222 | 3 / 28 / 1 | 32 | 0 | 32 |
| 0.70 | maskco | -8.90508 | 8 / 4 / 20 | 32 | 0 | 32 |
| 0.70 | recon_only | 0.00888 | 12 / 16 / 4 | 32 | 0 | 32 |
| 0.70 | discard | 0.00888 | 12 / 16 / 4 | 32 | 0 | 32 |
| 0.80 | maskco | -5.62654 | 12 / 0 / 20 | 32 | 0 | 32 |
| 0.80 | recon_only | -3.73916 | 12 / 4 / 16 | 32 | 0 | 32 |
| 0.80 | discard | -3.73916 | 12 / 4 / 16 | 32 | 0 | 32 |
| 0.90 | maskco | -5.62144 | 12 / 0 / 20 | 32 | 0 | 32 |
| 0.90 | recon_only | -7.49312 | 12 / 0 / 20 | 32 | 0 | 32 |
| 0.90 | discard | -7.49312 | 12 / 0 / 20 | 32 | 0 | 32 |
| 1.00 | maskco | -5.62144 | 12 / 0 / 20 | 32 | 0 | 32 |
| 1.00 | recon_only | -7.49312 | 12 / 0 / 20 | 32 | 0 | 32 |
| 1.00 | discard | -7.49312 | 12 / 0 / 20 | 32 | 0 | 32 |

## Fraction-by-fraction reading

### mask_fraction = 0.50

- recon_only identity rate **0.8125** (26/32 rows) -> control is INERT.
- recon_only row breakdown (n=32): 26 identical AND delta 0 (no change at all), 0 identical with nonzero delta, 2 different sequence but delta 0 (behaviourally blind), 4 different sequence AND nonzero delta (reached the outcome).
- mean raw delta utility: maskco -5.16080, recon_only 0.00222, discard 0.00222.
- paired `maskco - recon_only`: -9.39378, 95% descriptive bootstrap CI [-15.96824, -2.83070], n_pairs=16 (0 pos / 8 zero / 8 neg).
- the insertability guard `mask_extended_to_all_for_insertability` fired in 4 rows across 4 states.
- `discard == recon_only` control: paired mean 0.00000 over 16 pairs (must be exactly 0).

### mask_fraction = 0.70

- recon_only identity rate **0.3750** (12/32 rows) -> control is NON-INERT.
- recon_only row breakdown (n=32): 12 identical AND delta 0 (no change at all), 0 identical with nonzero delta, 4 different sequence but delta 0 (behaviourally blind), 16 different sequence AND nonzero delta (reached the outcome).
- mean raw delta utility: maskco -8.90508, recon_only 0.00888, discard 0.00888.
- paired `maskco - recon_only`: -14.09658, 95% descriptive bootstrap CI [-19.72060, -8.48096], n_pairs=16 (0 pos / 0 zero / 16 neg).
- the insertability guard `mask_extended_to_all_for_insertability` fired in 0 rows across 0 states.
- `discard == recon_only` control: paired mean 0.00000 over 16 pairs (must be exactly 0).

### mask_fraction = 0.80

- recon_only identity rate **0.1250** (4/32 rows) -> control is NON-INERT.
- recon_only row breakdown (n=32): 4 identical AND delta 0 (no change at all), 0 identical with nonzero delta, 0 different sequence but delta 0 (behaviourally blind), 28 different sequence AND nonzero delta (reached the outcome).
- mean raw delta utility: maskco -5.62654, recon_only -3.73916, discard -3.73916.
- paired `maskco - recon_only`: -0.04460, 95% descriptive bootstrap CI [-9.43291, 9.33387], n_pairs=16 (8 pos / 0 zero / 8 neg).
  The CI includes zero, so **THE DIFFERENCE IS NOT ESTABLISHED** at this fraction. That
  is not evidence of no difference.
- the insertability guard `mask_extended_to_all_for_insertability` fired in 0 rows across 0 states.
- `discard == recon_only` control: paired mean 0.00000 over 16 pairs (must be exactly 0).

### mask_fraction = 0.90

- recon_only identity rate **0.0000** (0/32 rows) -> control is NON-INERT.
- recon_only row breakdown (n=32): 0 identical AND delta 0 (no change at all), 0 identical with nonzero delta, 0 different sequence but delta 0 (behaviourally blind), 32 different sequence AND nonzero delta (reached the outcome).
- mean raw delta utility: maskco -5.62144, recon_only -7.49312, discard -7.49312.
- paired `maskco - recon_only`: 7.47352, 95% descriptive bootstrap CI [3.72251, 11.22944], n_pairs=16 (12 pos / 0 zero / 4 neg).
- the insertability guard `mask_extended_to_all_for_insertability` fired in 0 rows across 0 states.
- `discard == recon_only` control: paired mean 0.00000 over 16 pairs (must be exactly 0).

### mask_fraction = 1.00

- recon_only identity rate **0.0000** (0/32 rows) -> control is NON-INERT.
- recon_only row breakdown (n=32): 0 identical AND delta 0 (no change at all), 0 identical with nonzero delta, 0 different sequence but delta 0 (behaviourally blind), 32 different sequence AND nonzero delta (reached the outcome).
- mean raw delta utility: maskco -5.62144, recon_only -7.49312, discard -7.49312.
- paired `maskco - recon_only`: 7.47352, 95% descriptive bootstrap CI [3.72251, 11.22944], n_pairs=16 (12 pos / 0 zero / 4 neg).
- the insertability guard `mask_extended_to_all_for_insertability` fired in 0 rows across 0 states.
- `discard == recon_only` control: paired mean 0.00000 over 16 pairs (must be exactly 0).

## Per-state contrast (sign reversals must be visible, not averaged away)

### mask_fraction = 0.50

| inst | clock | \|S\| | maskco mean delta | recon_only mean delta | recon_only identity | maskco - recon_only (mean; +/-/0 over seeds) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 7.134 | 3 | -0.01175 | 3.73962 | 3/4 | -3.75136 (0/3/1) |
| 0 | 10.850 | 5 | -30.05655 | 0.00000 | 4/4 | -30.05655 (0/0/4) |
| 1 | 2.005 | 3 | -3.72727 | -3.74773 | 3/4 | 0.02046 (1/3/0) |
| 1 | 5.184 | 4 | -3.74576 | 0.00000 | 2/4 | -3.74576 (0/3/1) |
| 2 | 2.017 | 3 | 0.00235 | 0.00070 | 3/4 | 0.00166 (3/0/1) |
| 2 | 6.974 | 4 | -3.75050 | 0.00000 | 4/4 | -3.75050 (0/2/2) |
| 3 | 3.398 | 4 | 0.00017 | 0.00000 | 4/4 | 0.00017 (2/2/0) |
| 3 | 4.578 | 3 | 0.00288 | 0.02518 | 3/4 | -0.02230 (0/3/1) |

### mask_fraction = 0.70

| inst | clock | \|S\| | maskco mean delta | recon_only mean delta | recon_only identity | maskco - recon_only (mean; +/-/0 over seeds) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 7.134 | 3 | -0.04698 | 14.95847 | 0/4 | -15.00545 (0/0/4) |
| 0 | 10.850 | 5 | -30.06249 | 0.00000 | 4/4 | -30.06249 (0/0/4) |
| 1 | 2.005 | 3 | -14.90910 | -14.99093 | 0/4 | 0.08183 (4/0/0) |
| 1 | 5.184 | 4 | -11.23767 | 0.00000 | 0/4 | -11.23767 (0/0/4) |
| 2 | 2.017 | 3 | 0.00106 | 0.00278 | 0/4 | -0.00172 (0/0/4) |
| 2 | 6.974 | 4 | -14.99695 | 0.00000 | 4/4 | -14.99695 (0/0/4) |
| 3 | 3.398 | 4 | 0.00000 | 0.00000 | 4/4 | 0.00000 (0/4/0) |
| 3 | 4.578 | 3 | 0.01151 | 0.10070 | 0/4 | -0.08920 (0/0/4) |

### mask_fraction = 0.80

| inst | clock | \|S\| | maskco mean delta | recon_only mean delta | recon_only identity | maskco - recon_only (mean; +/-/0 over seeds) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 7.134 | 3 | -0.04698 | 14.95847 | 0/4 | -15.00545 (0/0/4) |
| 0 | 10.850 | 5 | -30.06249 | 0.00000 | 4/4 | -30.06249 (0/0/4) |
| 1 | 2.005 | 3 | -14.90910 | -14.99093 | 0/4 | 0.08183 (4/0/0) |
| 1 | 5.184 | 4 | -0.00162 | -14.98302 | 0/4 | 14.98140 (4/0/0) |
| 2 | 2.017 | 3 | 0.00106 | 0.00278 | 0/4 | -0.00172 (0/0/4) |
| 2 | 6.974 | 4 | -0.00507 | -14.99695 | 0/4 | 14.99188 (4/0/0) |
| 3 | 3.398 | 4 | 0.00033 | -0.00431 | 0/4 | 0.00464 (4/0/0) |
| 3 | 4.578 | 3 | 0.01151 | 0.10070 | 0/4 | -0.08920 (0/0/4) |

### mask_fraction = 0.90

| inst | clock | \|S\| | maskco mean delta | recon_only mean delta | recon_only identity | maskco - recon_only (mean; +/-/0 over seeds) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 7.134 | 3 | -0.04698 | 14.95847 | 0/4 | -15.00545 (0/0/4) |
| 0 | 10.850 | 5 | -30.02169 | -30.03170 | 0/4 | 0.01001 (4/0/0) |
| 1 | 2.005 | 3 | -14.90910 | -14.99093 | 0/4 | 0.08183 (4/0/0) |
| 1 | 5.184 | 4 | -0.00162 | -14.98302 | 0/4 | 14.98140 (4/0/0) |
| 2 | 2.017 | 3 | 0.00106 | 0.00278 | 0/4 | -0.00172 (0/0/4) |
| 2 | 6.974 | 4 | -0.00507 | -14.99695 | 0/4 | 14.99188 (4/0/0) |
| 3 | 3.398 | 4 | 0.00033 | -0.00431 | 0/4 | 0.00464 (4/0/0) |
| 3 | 4.578 | 3 | 0.01151 | 0.10070 | 0/4 | -0.08920 (0/0/4) |

### mask_fraction = 1.00

| inst | clock | \|S\| | maskco mean delta | recon_only mean delta | recon_only identity | maskco - recon_only (mean; +/-/0 over seeds) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 7.134 | 3 | -0.04698 | 14.95847 | 0/4 | -15.00545 (0/0/4) |
| 0 | 10.850 | 5 | -30.02169 | -30.03170 | 0/4 | 0.01001 (4/0/0) |
| 1 | 2.005 | 3 | -14.90910 | -14.99093 | 0/4 | 0.08183 (4/0/0) |
| 1 | 5.184 | 4 | -0.00162 | -14.98302 | 0/4 | 14.98140 (4/0/0) |
| 2 | 2.017 | 3 | 0.00106 | 0.00278 | 0/4 | -0.00172 (0/0/4) |
| 2 | 6.974 | 4 | -0.00507 | -14.99695 | 0/4 | 14.99188 (4/0/0) |
| 3 | 3.398 | 4 | 0.00033 | -0.00431 | 0/4 | 0.00464 (4/0/0) |
| 3 | 4.578 | 3 | 0.01151 | 0.10070 | 0/4 | -0.08920 (0/0/4) |

## Operationally equivalent fractions

These fraction pairs produced identical masked sets (identical per-state, per-mode
deltas), so they are NOT independent data points:

- mask_fraction in {0.90, 1.00} are operationally identical on this state set.

Cause: the mask size is `ceil(mask_fraction * |S|)`, which reaches `|S|` (mask
everything, empty retained route) as soon as `mask_fraction > 1 - 1/|S|`. For the
|S| in {3, 4, 5} this sweep uses, that threshold is 0.667/0.75/0.80, so 0.90 already
masks all orders in every state and 1.00 cannot differ from it. Reporting them as two
rows must not be read as replication.

## Which fraction produced a non-inert control

Fractions with a control identity rate at or below 0.5 (non-inert by the registered
definition), and what the contrast did there:

| mask fraction | recon_only identity rate | maskco - recon_only mean | 95% descriptive CI | CI includes zero | reading |
| --- | --- | --- | --- | --- | --- |
| 0.70 | 0.3750 | -14.09658 | [-19.72060, -8.48096] | False | point estimate negative, CI below zero (descriptive only) |
| 0.80 | 0.1250 | -0.04460 | [-9.43291, 9.33387] | True | difference NOT established |
| 0.90 | 0.0000 | 7.47352 | [3.72251, 11.22944] | False | point estimate positive, CI above zero (descriptive only) |
| 1.00 | 0.0000 | 7.47352 | [3.72251, 11.22944] | False | point estimate positive, CI above zero (descriptive only) |

Even where the control is non-inert, these are a handful of engineered states with no
`cond_hist` arm, so a non-inert control only makes the contrast *interpretable*, not
*confirmatory*.

## What the sweep shows about the guide

Paired `maskco - recon_only` point estimates by fraction: 0.50 -> -9.39378, 0.70 -> -14.09658, 0.80 -> -0.04460, 0.90 -> 7.47352, 1.00 -> 7.47352.

**The point estimate changes sign across fractions**: it is clearly negative at the
partial-mask fractions, indistinguishable from zero at 0.80, and positive once the
mask covers all orders. A single fraction does not summarise the guide, and picking
the favourable fraction would be selection. Taken as a whole, the sweep does not
establish a consistent direction.

Two further readings that constrain any interpretation:

- In every fraction the `maskco` mean raw delta is **negative**, so on average rebuilding
  a suffix here is worse than leaving the baseline suffix alone, guide or not. Where the
  contrast is positive it means the guided rebuild was *less damaging* than the
  distance-only rebuild in these rows; it does not mean the rebuild improved the day.
- The contrasts are driven by a few discrete states rather than by a broad shift: the
  observed deltas cluster at 0 and at about +/-3.75 and +/-15, i.e. they turn on whether
  a given permutation changed the accepted set or the hard-feasibility outcome. The
  per-state tables above show exactly which states drive each fraction, and the f=0.90
  mean is carried by two states where the distance-only rebuild cost about 15 each.

## Honest verdict on the guide

Within this sweep: the control was made non-inert (identity rate at or below 0.5) at
mask fraction 0.70, 0.80, 0.90, 1.00, so the contrast became *interpretable* there; but the contrast did not
hold a consistent sign or magnitude across those fractions, and a fraction whose CI includes zero (0.80) leaves the difference not established. The evidence that the guide
helps, hurts, or does nothing is therefore **undetermined** by this sweep. The following
scope statements hold regardless of the point estimates:

- These pilots have **no `cond_hist` arm** and few engineered states, so they **cannot
  address the pickup program MAIN gate**, which requires the new method minus `cond_hist`
  to be at least **20 utility/day** with a paired CI lower bound above 0, plus a positive
  mean versus the same-initial-value reconstruction-only control.
- A confidence interval that includes zero means **THE DIFFERENCE IS NOT ESTABLISHED**. It is
  **not** evidence of no difference, and it is **not** equivalence.
- Nothing here establishes a stable online gain, MaskCO attribution, pickup impossibility, or
  equivalence. No acceptance rule, no online policy and no registered panel is run.
- Producing a permutation is not producing a gain. The capture loop also favoured early
  events, and every failure and degenerate row is retained in `result.json` (see the
  failures column); nothing was redrawn or dropped.

## Files

- `D:\PyCharm_\MASKCO-Main\research\pickup_20round_audit\pickup_arm_pilot_mf50_v1\result.json` (mask_fraction 0.50, sha256 `c5114805b1eff6a5a6527d6bd70236265915049e34bc05f18b4f0ee9935252ba`)
- `D:\PyCharm_\MASKCO-Main\research\pickup_20round_audit\pickup_arm_pilot_mf70_v1\result.json` (mask_fraction 0.70, sha256 `bc398e321388053856eaaa33f5271ec7e8340aefb541991e4b0886cd2e3777c7`)
- `D:\PyCharm_\MASKCO-Main\research\pickup_20round_audit\pickup_arm_pilot_mf80_v1\result.json` (mask_fraction 0.80, sha256 `d365c1bce151bfb0438fb97a40e73ca2065fcf5d7315b900d7fb8653fbed7bc4`)
- `D:\PyCharm_\MASKCO-Main\research\pickup_20round_audit\pickup_arm_pilot_mf90_v1\result.json` (mask_fraction 0.90, sha256 `a2625087d3be2e4dded0a63be3f722cd358fa94c6835eb0f84b3402c162df188`)
- `D:\PyCharm_\MASKCO-Main\research\pickup_20round_audit\pickup_arm_pilot_mf100_v1\result.json` (mask_fraction 1.00, sha256 `547516278b939986383f6bb2d14015e3d397763f19999859f524a132615175d5`)
- `control_strength_sweep_summary.json` - machine-readable summary rendered into this file
