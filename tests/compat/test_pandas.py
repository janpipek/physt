import numpy as np
import pytest

pd = pytest.importorskip("pandas")


@pytest.fixture
def series_of_int() -> pd.Series:
    return pd.Series([0, 1, 2, 3, 4, 5])


@pytest.fixture
def series_of_nullable_int() -> pd.Series:
    return pd.Series([0, 1, 2, 3, pd.NA, 4], dtype="Int64")


@pytest.fixture
def series_of_str() -> pd.Series:
    return pd.Series(["a", "b", "c"], dtype="string")


@pytest.fixture
def df_one_column(series_of_int: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
    {
        "a": series_of_int
    })


@pytest.fixture
def df_two_columns(series_of_int: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
    {
        "a": series_of_int,
        "b": series_of_int * 2,
    })


@pytest.fixture
def df_three_columns(series_of_int: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
    {
        "a": series_of_int,
        "b": series_of_int * 2,
    })


class TestBinningToIndex:
    pass


class TestPhystSeriesAccessors:
    def test_exists_compatible_dtype(self, series_of_int) -> None:
        assert hasattr(series_of_int, "physt")

    def test_does_not_exists_incompatible_dtype(self, series_of_str) -> None:
        assert not hasattr(series_of_str, "physt")

    class TestH1:
        # TODO: Just check that it works
        pass



class TestPhystDataFrameAccessors:
    def test_exists(self, df_one_column: pd.DataFrame):
        assert hasattr(df_one_column, "physt")

    class TestH2:
        # TODO: Just check that it works
        pass


class TestH1ToSeries:
    pass


class TestH1ToDDataFrame:
    pass