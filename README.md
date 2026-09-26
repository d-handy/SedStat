<a id="top"></a>

<!-- ABOUT SedStat -->
## About SedStat

Grain size distribution analysis for laser granulometer data and classical sieve/pipette lab workflows. All five statistical methods from Blott & Pye (2001) are implemented: arithmetic, geometric, and logarithmic moments, plus graphical statistics in logarithmic and geometric form. These feed percentiles, Wentworth-scale grain-size fractions, and textual classifications for sorting, skewness, kurtosis, and grain-size class, forming the basis for texture analysis.

Plotting samples on a sand-silt-clay ternary diagram, as overlaid cumulative curves, or as a stratigraphic heatmap against depth is planned, as is a desktop application that wraps the same statistics.

**Current state.** `sedstat.core` (statistics, classification, fractions, percentiles, Stokes' law, bimodality) and `sedstat.io` (Beckman Coulter LS `.$av` files, sieve/pipette CSV) are usable today from Python. `sedstat.plotting` and `sedstat.gui` are not written yet, so anything involving figures or the desktop application does not work in this version.

<p align="right">(<a href="#top">top</a>)</p>

## Installation

Requires Python ≥ 3.11.

```bash
# Core library and file readers
pip install -e .

# Also pull in the GUI and plotting dependencies (PySide6, matplotlib,
# seaborn, pandas). The modules that use them are still being written.
pip install -e ".[gui]"
```

<p align="right">(<a href="#top">top</a>)</p>


### Development


```bash
uv sync --extra gui --dev
uv run pytest tests/test_statistics.py tests/test_classification.py tests/test_fractions.py \
               tests/test_stokes.py tests/test_reference_distributions.py \
               tests/test_io_beckman_coulter.py tests/test_io_sieve_pipette.py -q
```

The full `uv run pytest tests/ -q` aborts during collection. The test suite
covers the whole planned package, so ten of its files import `sedstat.gui`
or `sedstat.plotting`, which do not exist yet.

<p align="right">(<a href="#top">top</a>)</p>

---

## Usage

### Python API

Sieve and pipette weights go in a CSV with one row per weighed fraction,
grouped by sample. The pan row carries `mesh_um=0`; pipette rows give either
a diameter in `size_um` or the raw settling conditions, from which Stokes'
law resolves one:

```csv
group_id,type,mesh_um,weight_g,size_um,time_s,depth_cm,temp_c,density_g_cm3
S1,sieve,0,6.0,,,,,
S1,sieve,63,20.0,,,,,
S1,sieve,250,50.0,,,,,
S1,pipette,,2.0,2,,,,
S1,pipette,,3.0,20,,,,
```

Reading it gives one record per sample, ready for `compute_all`:

```python
from sedstat.core import compute_all
from sedstat.io import read_sieve_pipette_csv

for record in read_sieve_pipette_csv("sample.csv"):
    result = compute_all(record.classes_um, record.values)
    print(record.group_id, record.method)
    print(f"  mean        {result.folk_ward_log.mean:.2f} phi")
    print(f"  sorting     {result.descriptions['fw_log_sorting']}")
    print(f"  D50         {result.percentiles.p50:.1f} um")
    print(f"  sand/silt   {result.fractions.total_sand:.1f} / {result.fractions.total_silt:.1f} %")
```

```text
S1 combined
  mean        4.82 phi
  sorting     Very Poorly Sorted
  D50         48.4 um
  sand/silt   76.9 / 23.1 %
```

Laser granulometer exports are the other way in, and produce the same
`(classes_um, values)` pair:

```python
from sedstat.io import read_ls_file

record = read_ls_file("sample.$av")
result = compute_all(record.classes_um, record.values)
```

`result` also carries the arithmetic, geometric, and logarithmic moment
statistics, both Folk and Ward variants, the full percentile set, and the
Wentworth fractions. `result.to_dict()` flattens all of it into one row.

The classification scheme is selectable: `compute_all(..., scheme=ClassificationScheme.USDA)`
moves the silt/sand boundary to 50 µm, and `ISO14688` adds that standard's
subdivisions of the coarse end.

<p align="right">(<a href="#top">top</a>)</p>

### Desktop application

Not available yet. Once `sedstat.gui` lands, the installed `sedstat` command
starts it.

<p align="right">(<a href="#top">top</a>)</p>


---

## Reference

Blott, S.J. & Pye, K. (2001). GRADISTAT: a grain size distribution and statistics
package for the analysis of unconsolidated sediments.
*Earth Surface Processes and Landforms*, 26, 1237–1248.
<https://doi.org/10.1002/esp.261>

Dietze, M., Schulte, P. and Dietze, E. (2022), Application of end-member modelling to grain-size data: Constraints and limitations. Sedimentology, 69: 845-863. https://doi.org/10.1111/sed.12929

Further sources for the individual methods are listed in [REFERENCES.md](REFERENCES.md).
