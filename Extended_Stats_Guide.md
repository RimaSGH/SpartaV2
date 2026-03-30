# Extended Statistics Guide

This guide covers the 9 new gap-length summary statistics added in the `extended-stats` branch,
how to use them during inference, and how to add your own.

---

## The 9 New Statistics

All statistics operate on the **unique gap lengths** extracted from an MSA — the same
gap positions that the C++ `msastats` library uses internally.

| Index | Name | Description |
|-------|------|-------------|
| SS_27 | `UNIQUE_GAP_Q25` | 25th percentile of unique gap lengths |
| SS_28 | `UNIQUE_GAP_Q50` | 50th percentile (median) |
| SS_29 | `UNIQUE_GAP_Q75` | 75th percentile |
| SS_30 | `UNIQUE_GAP_Q95` | 95th percentile |
| SS_31 | `GAP_VARIANCE` | Variance of unique gap lengths |
| SS_32 | `GAP_CV` | Coefficient of variation (σ/μ) |
| SS_33 | `GAP_SHANNON_ENTROPY` | Shannon entropy of gap-length distribution |
| SS_34 | `GAP_MAX_TO_MEAN` | Max gap / mean gap ratio |
| SS_35 | `GAP_DENSITY_RATIO` | Proportion of alignment positions that are gaps |

By default all 9 are enabled and included in every run alongside SS_0–SS_26.

---

## Using the `-s` Flag to Select Stat Subsets

The `-s` flag passed to `sparta-abc` lists statistics to **exclude**.
This lets you test any combination without re-running simulations.

```bash
# Use all 36 statistics (default — no -s flag)
sparta-abc -i <input_dir> -noc

# Use only the 9 new statistics (exclude all 27 C++ base stats)
sparta-abc -i <input_dir> -noc -s SS_0 SS_1 SS_2 SS_3 SS_4 SS_5 SS_6 SS_7 SS_8 SS_9 \
    SS_10 SS_11 SS_12 SS_13 SS_14 SS_15 SS_16 SS_17 SS_18 SS_19 \
    SS_20 SS_21 SS_22 SS_23 SS_24 SS_25 SS_26

# Use only Q25 + Q50 + Q75 (best combination found in benchmarks)
sparta-abc -i <input_dir> -noc -s SS_0 SS_1 SS_2 SS_3 SS_4 SS_5 SS_6 SS_7 SS_8 SS_9 \
    SS_10 SS_11 SS_12 SS_13 SS_14 SS_15 SS_16 SS_17 SS_18 SS_19 \
    SS_20 SS_21 SS_22 SS_23 SS_24 SS_25 SS_26 SS_30 SS_31 SS_32 SS_33 SS_34 SS_35

# Exclude only GAP_CV (SS_32) — known to hurt inference
sparta-abc -i <input_dir> -noc -s SS_32
```

---

## How to Add Your Own Statistic

Open `spartaabc/ext_stats.py` and add a decorated function. That is all that is needed —
the registry, simulation pipeline, and inference pipeline all pick it up automatically.

### Step 1: Write the function in `ext_stats.py`

```python
from spartaabc.stat_registry import registry
import numpy as np

@registry.register(
    name="MY_STAT_NAME",
    description="One sentence describing what this measures",
    category="custom",
    enabled=True          # set False to register but not use by default
)
def calculate_my_stat(gap_lengths: np.ndarray) -> float:
    if len(gap_lengths) == 0:
        return 0.0
    # your calculation here
    return float(np.median(gap_lengths))
```

The function receives a 1D `numpy` array of unique gap lengths (one value per unique
alignment position) and must return a single `float`.

### Step 2: Reinstall the package

```bash
cd SpartaV2
pip install -e .
```

### Step 3: Verify the stat is registered

```bash
python3 -c "
from spartaabc import ext_stats
from spartaabc.stat_registry import registry
print(registry.list_stats(enabled_only=True))
"
```

Your stat name should appear in the list. It will be assigned the next available SS index
(e.g. SS_36 if all 9 existing stats are enabled).

### Step 4: Re-run simulations

The stat is computed during simulation and stored in the parquet. If you have existing
parquet files, they do not contain your new stat — you must re-run `sparta-simulate`.

```bash
sparta-simulate -i <input_dir> -n 100000 -m sim -p prior.json
```

After that, `sparta-abc` will use your stat automatically.

---

## Disabling a Statistic Without Removing It

Set `enabled=False` in the decorator to keep the function registered but exclude it
from all calculations. This is useful for stats you want to test selectively
using the `-s` flag without breaking the index ordering of other stats.

```python
@registry.register(name="MY_STAT", enabled=False)
def calculate_my_stat(gap_lengths):
    ...
```

---

## Note on `GAP_DENSITY_RATIO`

`GAP_DENSITY_RATIO` (SS_35) is a special case. Its `ext_stats.py` function returns 0.0
as a placeholder because it requires access to the full MSA matrix, not just the gap
lengths array. The real value is computed separately in `simulate_data.py` and
`stats_manager.py` via `external_utils.compute_gap_density(sequences)` and injected
after the registry call. If you add a similar statistic that needs full MSA access,
follow the same pattern.
