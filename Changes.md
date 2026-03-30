# Changes from Original SpartaV2

This branch extends the original [SpartaV2](https://github.com/elyawy/SpartaV2) with 9 new Python-based
gap-length summary statistics (SS_27–SS_35) on top of the existing 27 C++ statistics (SS_0–SS_26).

---

## New Files

### `spartaabc/ext_stats.py`
Implements the 9 new statistics using the `@registry.register` decorator.
Each function takes a `np.ndarray` of unique gap lengths and returns a single `float`.
Also contains one disabled example statistic (`GAP_LONG_RATIO`) to demonstrate how users can add their own.

### `spartaabc/stat_registry.py`
A decorator-based registry system (`StatRegistry` class) for managing extended statistics.
Handles registration, enabling/disabling, and batch calculation of all registered stats.
A global `registry` instance is shared across the package.

### `spartaabc/external_utils.py`
Python reimplementation of the C++ `fillUniqueGapsMap()` logic from `msastats`.
Provides `get_unique_gap_lengths(msa)` and `compute_gap_density(msa)` — the data bridge
between raw MSA sequences and the extended statistics.

### `spartaabc/stats_manager.py`
Coordinates C++ base statistics with Python extended statistics.
Exposes `calculate_all_extended_stats(sequences)` — the main entry point used by
`abc_inference.py` and `simulate_data.py`.

---

## Modified Files

### `spartaabc/utility.py`
Two changes:

1. **Extended `SUMSTATS_LIST` from 27 to 36 entries.**
   Imports the registry at module load time and appends SS_27–SS_35 dynamically.
   This makes the rest of the pipeline aware of the new statistics without hardcoding.

2. **Fixed macOS hidden-file crash in `get_msa_path()` and tree path detection.**
   The original code used `glob("*.fasta") == 1` which fails when macOS creates
   hidden `._filename` ghost files alongside real files (common after `scp` transfers).
   The fix filters out any file whose name starts with `._`.

### `spartaabc/simulate_data.py`
After computing the 27 C++ statistics per simulated MSA, now also computes all 9 extended
statistics and appends them to each row before writing to the parquet file.
The output parquet therefore has 36 statistic columns (SS_0–SS_35) instead of 27.

### `spartaabc/abc_inference.py`
After computing the 27 C++ statistics on the empirical (observed) MSA, now also computes
the 9 extended statistics — both in the MAFFT-realigned path and the no-correction (`-noc`) path.
This ensures the empirical stat vector and the simulation stat vectors have the same dimensionality.

---

## What Did Not Change

- `pyproject.toml` — identical to original; entry points unchanged
- `main.py`, `aligner_interface.py`, `correction.py`, `prior_sampler.py`, etc. — untouched
- The `-s` flag for excluding statistics was already present in the original `abc_inference.py`
  and continues to work with the extended stat indices (e.g. `-s SS_27 SS_28` to exclude new stats)
