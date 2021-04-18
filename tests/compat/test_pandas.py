import pytest

pd = pytest.importorskip("pandas")


@pytest.fixture()
def series() -> pd.Series:
    return pd.Series([0, 1, 2, 3, 4, 5])


@pytest.fixture()
def series_nullable_int() -> pd.Series:
    return pd.Series([0, 1, 2, 3, pd.NA, 4], dtype="Int64")


class TestBinningToIndex:
    pass


class TestPhystSeriesAccessors:
    def test_exists(self, series: pd.Series) -> None:
        assert hasattr(series, "physt")

class TestH1ToSeries:
    pass


class TestH1ToDDataFrame:
    pass