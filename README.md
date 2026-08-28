<!-- ABOUT SedStat -->
## About SedStat

Grain size distribution analysis for laser granulometer data and classical sieve/pipette lab workflows. All five statistical methods from Blott & Pye (2001) are implemented: arithmetic, geometric, and logarithmic moments, plus graphical statistics in logarithmic and geometric form. These feed percentiles, Wentworth-scale grain-size fractions, and textual classifications for sorting, skewness, kurtosis, and grain-size class, forming the basis for texture analysis.

Samples can be plotted on sand-silt-clay ternary diagram, compared as overlaid cumulative curves, or laid out as a stratigraphic heatmap against depth. The same statistics and plots are available from a desktop GUI or a Python API.

<p align="right">(<a href="#top">top</a>)</p>

## Installation

Requires Python ≥ 3.11.

```bash
# Core library only (no GUI, no plots)
pip install -e .

# With GUI and plotting
pip install -e ".[gui]"
```

<p align="right">(<a href="#top">top</a>)</p>


### Development


```bash
uv sync --extra gui --dev
uv run pytest tests/ -q
```
<p align="right">(<a href="#top">top</a>)</p>

---

## Usage

### Desktop application

```bash
sedstat
```

Or from the source tree:

```bash
python -m sedstat.gui.app
```

<p align="right">(<a href="#top">top</a>)</p>


---

## Reference

Blott, S.J. & Pye, K. (2001). GRADISTAT: a grain size distribution and statistics
package for the analysis of unconsolidated sediments.
*Earth Surface Processes and Landforms*, 26, 1237–1248.
<https://doi.org/10.1002/esp.261>
