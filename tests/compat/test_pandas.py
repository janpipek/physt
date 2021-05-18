from typing import Any, Dict, Iterable

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
from pandas import IntervalIndex
from pandas.testing import assert_index_equal
from physt import h1, h2
from physt.binnings import static_binning
from physt.compat.pandas import binning_to_index
from physt.testing import assert_histograms_equal


@pytest.fixture
def series_of_int() -> pd.Series:
    return pd.Series([0, 1, 2, 3, 4, 5], name="series_of_int")


@pytest.fixture
def series_of_nullable_int() -> pd.Series:
    return pd.Series([0, 1, 2, 3, pd.NA, 4], dtype="Int64")


@pytest.fixture
def series_of_str() -> pd.Series:
    return pd.Series(["a", "b", "c"], dtype="string")


@pytest.fixture
def df_one_column(series_of_int: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"a": series_of_int})


@pytest.fixture
def df_two_columns(series_of_int: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": series_of_int,
            "b": series_of_int * 2,
        }
    )


@pytest.fixture
def df_three_columns(series_of_int: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": series_of_int,
            "b": series_of_int * 2,
            "c": series_of_int * 3,
        }
    )


@pytest.fixture
def df_multilevel_column_index(series_of_int: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            ("a", "b"): series_of_int,
            ("a", "c"): series_of_int * 2,
            ("b", "d"): series_of_int * 3,
        }
    )


class TestBinningToIndex:
    def test_binning(self):
        binning = static_binning(data=None, bins=[1, 2, 3, 4])
        index = binning_to_index(binning)
        expected = IntervalIndex.from_breaks(
            [1, 2, 3, 4], closed="left", dtype="interval[int64]"
        )
        assert_index_equal(index, expected)


class TestPhystSeriesAccessors:
    def test_exists_compatible_dtype(self, series_of_int) -> None:
        assert hasattr(series_of_int, "physt")
        assert hasattr(series_of_int.physt, "h1")
        assert not hasattr(series_of_int.physt, "h2")

    def test_does_not_exist_incompatible_dtype(self, series_of_str) -> None:
        assert not hasattr(series_of_str, "physt")

    class TestH1:
        @pytest.mark.parametrize(
            "args,kwargs",
            [([], {}), (["human"], {}), (["fixed_width"], {"bin_width": 0.2})],
        )
        def test_same_as_array(
            self, series_of_int: pd.Series, args: Iterable[Any], kwargs: Dict[str, Any]
        ):
            output = series_of_int.physt.h1(*args, **kwargs)
            expected = h1(np.asarray(series_of_int.array), *args, **kwargs)
            assert_histograms_equal(output, expected, check_metadata=False)
            assert output.name == "series_of_int"

        @pytest.mark.parametrize(
            "args,kwargs",
            [
                ([], {"dropna": True}),
            ],
        )
        def test_same_as_array_na(
            self,
            series_of_nullable_int: pd.Series,
            args: Iterable[Any],
            kwargs: Dict[str, Any],
        ):
            output = series_of_nullable_int.physt.h1(*args, **kwargs)
            expected = h1(np.array([0, 1, 2, 3, 4]), *args, **kwargs)
            assert_histograms_equal(output, expected, check_metadata=False)

        def test_raise_nullable_with_no_dropna(
            self, series_of_nullable_int: pd.Series
        ) -> None:
            with pytest.raises(ValueError, match="Cannot histogram series with NA's"):
                series_of_nullable_int.physt.h1(dropna=False)


class TestPhystDataFrameAccessors:
    def test_exists(self, df_one_column: pd.DataFrame):
        assert hasattr(df_one_column, "physt")

    class TestH1:
        @pytest.mark.parametrize(
            "args,kwargs",
            [([], {}), (["human"], {}), (["fixed_width"], {"bin_width": 0.2})],
        )
        def test_single_column(self, df_one_column: pd.DataFrame, args: Iterable[Any], kwargs: Dict[str, Any]) -> None:
            # TODO: Test no argument separately
            # Non-trivial *args should perhaps fail?
            output = df_one_column.physt.h1(None, *args, **kwargs)
            expected = h1(np.asarray(df_one_column["a"].array), *args, **kwargs)
            assert_histograms_equal(output, expected, check_metadata=False)
            assert output.name == "a"

        def test_two_columns_no_arg(self, df_two_columns: pd.DataFrame) -> None:
            with pytest.raises(ValueError, match="Argument `column` must be set"):
                output = df_two_columns.physt.h1()

        @pytest.mark.parametrize(
            "index",
            [("a", "b"), ("a", "c"), ("non", "existent"), "a", ("a",)]
        )
        def test_with_multilevel_index(self, df_multilevel_column_index: pd.DataFrame, index: Any) -> None:
            try:
                data = df_multilevel_column_index[index]
            except:
                with pytest.raises(KeyError):
                    df_multilevel_column_index.physt.h1(index)
            else:
                if not isinstance(data, pd.Series):
                    with pytest.raises(ValueError, match="Argument `column` must select a single series"):
                        df_multilevel_column_index.physt.h1(index)
                else:
                    output = df_multilevel_column_index.physt.h1(index)
                    expected = h1(data)

    class TestH2:
        # TODO: Just check that it works
        pass

    class TestH3:
        pass


class TestH1ToSeries:
    pass


class TestH1ToDDataFrame:
    pass
