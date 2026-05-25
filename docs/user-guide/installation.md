# Installation

## Requirements

- Python 3.12 or later

## Install with uv

```bash
uv add lazymerge
```

## Install with pip

```bash
pip install lazymerge
```

## Optional dependencies

**Icechunk** -- if your Zarr data is stored in an 
[Icechunk](https://icechunk.io) repository, install `icechunk` separately
currently we only support icechunk==1.1.21 (due to how sessions are serialized
for use with Zarrs in zarr-datafusion-search:

```bash
uv add icechunk==1.1.21
```

lazymerge will automatically detect Icechunk stores and use the appropriate
constructors when querying sources via DataFusion.

## Running the example notebooks

The example notebooks live in `docs/examples/` and have their own
`pyproject.toml` with additional dependencies (xarray, matplotlib, icechunk,
etc.). To run them:

```bash
cd docs/examples
uv sync
uv run --with jupyter jupyter lab
```

This installs the example dependencies (including lazymerge as an editable
local package) and launches JupyterLab. The available notebooks are:

- **conventions.ipynb** -- working with spatial/projection conventions and
  overview pyramids
- **datafusion.ipynb** -- SQL-based source discovery with DataFusion
- **stac_virtualizarr.ipynb** -- ingesting STAC items as VirtualiZarr
  references and merging them
