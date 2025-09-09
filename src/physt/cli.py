from importlib.util import find_spec
from pathlib import Path

import click
import narwhals as nw
import rich

from physt._facade import h1


@click.group()
def app():
    pass


@app.command(name="h1")
@click.argument("path", type=click.Path(file_okay=True, dir_okay=False, path_type=Path))
@click.option(
    "-c", "--column", type=str, required=True, help="Name of the column to use"
)
@click.option("--dropna", is_flag=True)
def h1_(path: Path, column: str, dropna: bool):
    """Print a 1D histogram of data from a file"""
    data = _load_data(path).to_native()
    h = h1(data[column], dropna=dropna)
    rich.print(h)
    rich.print(h.statistics)
    h.plot.hbar(backend="ascii", show_values=True)


def _load_data(path: Path) -> nw.DataFrame:
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


def _load_data_pandas(path: Path):

    import pandas as pd

    match path.suffix:
        case ".csv":
            return pd.read_csv(path)
        case ".json":
            return pd.read_json(path)
        case _:
            raise ValueError(f"Unsupported file format: {path.suffix}")


def _load_data_polars(path: Path):
    import polars as pl

    return pl.read_csv(path)


if __name__ == "__main__":
    app()
