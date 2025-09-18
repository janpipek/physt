"""The `physt` CLI command.

It is in an early stage of development.
"""

from importlib.util import find_spec
from pathlib import Path
from typing import Any

import click
import narwhals as nw
import rich

from physt._facade import h1
from physt.histogram1d import Histogram1D


@click.group()
def app():
    pass


@app.command(name="h1")
@click.argument("path", type=click.Path(file_okay=True, dir_okay=False, path_type=Path))
@click.option(
    "-c", "--column", type=str, required=True, help="Name of the column to use"
)
@click.option("-n", "--bin-count", type=int, help="(Approximate) number of bins to use")
@click.option("-w", "--bin-width", type=float, help="Explicitly set bin width")
@click.option(
    "-p", "--pretty", is_flag=True, help="Make the bin width pretty and rounded"
)
@click.option("--dropna", is_flag=True, help="Ignore missing values")
@click.option("--json", is_flag=True, help="Print JSON representation of the histogram")
def h1_(*, path: Path, column: str, json: bool, **kwargs):
    """Print a 1D histogram of data from a file."""
    data = _load_data(path).to_native()

    # Get the histogram
    hist_kwargs = _extract_h1_kwargs(kwargs)
    try:
        col = data[column]
    except KeyError:
        rich.print(f"Column '{column}' not found in the data.")
        rich.print(f"Available columns: {', '.join(data.columns)}")
        exit(-1)
    h = h1(col, **hist_kwargs)

    # Output
    if json:
        print(h.to_json())
    else:
        _print_stats(h)
        h.plot.hbar(backend="ascii", show_values=True, show_labels=True)


@app.command()
def examples():
    """Show some of the physt examples."""
    from physt.examples import show_examples

    show_examples()


def _load_data(path: Path) -> nw.DataFrame:
    """Load data from a file.

    This tries both pandas and polars backends (and various formats).
    """
    supported_backends = ["pandas", "polars"]  # TODO: revert
    available_backends = (
        backend for backend in supported_backends if find_spec(backend)
    )

    if not available_backends:
        raise ValueError("No supported backend found")

    for backend in available_backends:
        try:
            return nw.read_csv(str(path), backend=backend)
        except KeyError:
            try:
                return nw.read_parquet(str(path), backend=backend)
            except EnvironmentError:
                continue

    raise ValueError(f"Unsupported file format: {path}")


def _extract_h1_kwargs(kwargs) -> dict[str, Any]:
    """Find appropriate keyword arguments for histogram creation.

    It's not the most elegant solution, but it works :-(
    """

    pass_directly = ["dropna"]
    hist_kwargs = {
        key: value
        for key in pass_directly
        if (value := kwargs.pop(key, None)) is not None
    }
    pretty = kwargs.pop("pretty", False)
    if bin_count := kwargs.pop("bin_count", None) or pretty:
        hist_kwargs["bins"] = "pretty" if pretty else "numpy"
        hist_kwargs["bin_count"] = bin_count
    if bin_width := kwargs.pop("bin_width", None):
        hist_kwargs["bins"] = "fixed_width"
        hist_kwargs["bin_width"] = bin_width
    return hist_kwargs


def _print_stats(h1: Histogram1D):
    rich.print(h1.name or h1.title or h1.axis_names[0] or "Histogram")
    statistics = h1.statistics
    rich.print(f"  Total: {h1.total}")
    rich.print(f"  Mean: {statistics.mean}")
    rich.print(f"  Median: {statistics.median}")
    rich.print(f"  Standard Deviation: {statistics.std}")
    rich.print()


if __name__ == "__main__":
    app()
