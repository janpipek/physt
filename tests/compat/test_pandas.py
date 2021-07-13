from typing import Any, Dict, Iterable

import numpy as np
import pytest

from physt.facade import h
from physt.histogram1d import Histogram1D
from physt.histogram_nd import HistogramND, Histogram2D

pytest.importorskip("pandas")
import pandas as pd
from pandas import IntervalIndex
from pandas.testing import assert_index_equal, assert_series_equal, assert_frame_equal
from physt import h1, h2
from physt.binnings import static_binning, StaticBinning
from physt.compat.pandas import binning_to_index, index_to_binning
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
def df_with_str(series_of_int: pd.Series, series_of_str: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"int": series_of_int, "str": series_of_str})


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
        expected = IntervalIndex.from_breaks([1, 2, 3, 4], closed="left", dtype="interval[int64]")
        assert_index_equal(index, expected)


class TestIndexToBinning:
    def test_valid_index(self) -> None:
        index = IntervalIndex.from_breaks(
            [0, 1, 1.5, 2, 3], closed="left", dtype="interval[float64]", name="Name"
        )
        output = index_to_binning(index)
        expected = static_binning(bins=[0, 1, 1.5, 2, 3])
        assert output == expected

    def test_non_rising_index(self) -> None:
        index = IntervalIndex.from_arrays(left=[1, 0], right=[2, 1], closed="left")
        with pytest.raises(ValueError, match="Bins must be in rising order."):
            index_to_binning(index)

    def test_overlapping_index(self) -> None:
        index = IntervalIndex.from_arrays(left=[0, 0.8], right=[1, 1.5], closed="left")
        with pytest.raises(ValueError, match="Intervals cannot overlap"):
            index_to_binning(index)

    def test_right_closed_index(self) -> None:
        index = IntervalIndex.from_arrays(
            left=[0, 1, 2, 1], right=[0.5, 1.5, 2.5, 1.5], closed="right"
        )
        with pytest.raises(ValueError, match="Only `closed_left` indices supported"):
            index_to_binning(index)


class TestPhystSeriesAccessors:
    def test_exists_compatible_dtype(self, series_of_int) -> None:
        assert hasattr(series_of_int, "physt")
        assert hasattr(series_of_int.physt, "h1")
        assert hasattr(series_of_int.physt, "histogram")
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

        def test_raise_nullable_with_no_dropna(self, series_of_nullable_int: pd.Series) -> None:
            with pytest.raises(ValueError, match="Cannot histogram series with NA's"):
                series_of_nullable_int.physt.h1(dropna=False)


class TestPhystDataFrameAccessors:
    def test_exists(self, df_one_column: pd.DataFrame) -> None:
        assert hasattr(df_one_column, "physt")
        assert hasattr(df_one_column.physt, "h1")
        assert hasattr(df_one_column.physt, "h2")
        assert hasattr(df_one_column.physt, "histogram")

    # TODO: Test weights

    class TestH1:
        @pytest.mark.parametrize(
            "args,kwargs",
            [([], {}), (["human"], {}), (["fixed_width"], {"bin_width": 0.2})],
        )
        def test_single_column(
            self,
            df_one_column: pd.DataFrame,
            args: Iterable[Any],
            kwargs: Dict[str, Any],
        ) -> None:
            # TODO: Test no argument separately
            # Non-trivial *args should perhaps fail?
            output: Histogram1D = df_one_column.physt.h1(None, *args, **kwargs)
            expected: Histogram1D = h1(np.asarray(df_one_column["a"].array), *args, **kwargs)
            assert_histograms_equal(output, expected, check_metadata=False)

        def test_correct_meta_data(self, df_one_column: pd.DataFrame) -> None:
            output: Histogram1D = df_one_column.physt.h1()
            assert output.name == "a"
            assert output.axis_name == "a"

        def test_two_columns_no_arg(self, df_two_columns: pd.DataFrame) -> None:
            with pytest.raises(ValueError, match="Argument `column` must be set"):
                df_two_columns.physt.h1()

        def test_invalid_dtype(self, df_with_str: pd.DataFrame) -> None:
            with pytest.raises(ValueError, match="Column 'str' is not numeric"):
                df_with_str.physt.h1("str")

        @pytest.mark.parametrize(
            "column_name,bin_arg",
            [("a", None), ("a", "human"), ("b", None), ("xxx", None)],
        )
        def test_two_columns(
            self, df_two_columns: pd.DataFrame, column_name: str, bin_arg: Any
        ) -> None:
            try:
                data = df_two_columns[column_name]
            except:
                with pytest.raises(KeyError, match=f"Column '{column_name}' not found"):
                    df_two_columns.physt.h1(column_name, bin_arg)
            else:
                expected = h1(data, bin_arg)
                output = df_two_columns.physt.h1(column_name, bin_arg)
                assert_histograms_equal(output, expected, check_metadata=False)

        @pytest.mark.parametrize(
            "index", [("a", "b"), ("a", "c"), ("non", "existent"), "a", ("a",)]
        )
        def test_with_multilevel_index(
            self, df_multilevel_column_index: pd.DataFrame, index: Any
        ) -> None:
            try:
                data = df_multilevel_column_index[index]
            except:
                with pytest.raises(KeyError):
                    df_multilevel_column_index.physt.h1(index)
            else:
                if not isinstance(data, pd.Series):
                    with pytest.raises(
                        ValueError,
                        match="Argument `column` must select a single series",
                    ):
                        df_multilevel_column_index.physt.h1(index)
                else:
                    output = df_multilevel_column_index.physt.h1(index)
                    expected = h1(data)
                    assert_histograms_equal(output, expected, check_metadata=False)

    class TestH2:
        def test_one_column(self, df_one_column: pd.DataFrame) -> None:
            with pytest.raises(ValueError, match="At least two columns required for 2D histograms"):
                df_one_column.physt.h2()

        def test_two_columns(self, df_two_columns: pd.DataFrame) -> None:
            output = df_two_columns.physt.h2()
            expected = h2(df_two_columns["a"], df_two_columns["b"])

        def test_three_columns_no_arg(self, df_three_columns: pd.DataFrame) -> None:
            with pytest.raises(ValueError, match="Arguments `column1` and `column2` must be set"):
                output = df_three_columns.physt.h2()

        def test_invalid_dtype(self, df_with_str: pd.DataFrame) -> None:
            with pytest.raises(ValueError, match="Column 'str' is not numeric"):
                df_with_str.physt.h2()

        def test_meta_data(
            self, df_two_columns: pd.DataFrame, df_three_columns: pd.DataFrame
        ) -> None:
            h = df_three_columns.physt.h2(column1="a", column2="b")
            assert h.name is None  # TODO: Do we now how to call it?
            assert h.axis_names == ("a", "b")

            h = df_two_columns.physt.h2()
            assert h.axis_names == ("a", "b")

            h = df_two_columns.physt.h2(axis_names=("b", "c"))
            assert h.axis_names == ("b", "c")

    class TestHistogram:
        def test_no_args(self, df_three_columns: pd.DataFrame) -> None:
            output: HistogramND = df_three_columns.physt.histogram()
            expected = h(df_three_columns.values)
            assert_histograms_equal(output, expected, check_metadata=False)

        @pytest.mark.parametrize("columns_arg", [["a", "b", "d"], [0, 1]])
        def test_invalid_columns(self, df_three_columns: pd.DataFrame, columns_arg: Any) -> None:
            with pytest.raises(KeyError, match="At least one of the columns .+ could not be found"):
                df_three_columns.physt.histogram(columns=columns_arg)

        def test_with_no_columns(self, df_three_columns) -> None:
            with pytest.raises(
                ValueError, match="Cannot make histogram from DataFrame with no columns"
            ):
                df_three_columns.physt.histogram(columns=[])

        def test_invalid_dtype(self, df_with_str: pd.DataFrame) -> None:
            with pytest.raises(ValueError, match="Column 'str' is not numeric"):
                df_with_str.physt.histogram()

        def test_scalar_columns_arg(self, df_three_columns: pd.DataFrame) -> None:
            output = df_three_columns.physt.histogram(columns="a")
            assert isinstance(output, Histogram1D)

        def test_meta_data(self, df_three_columns: pd.DataFrame) -> None:
            output: HistogramND = df_three_columns.physt.histogram()
            assert output.axis_names == ("a", "b", "c")

        def test_two_columns(self, df_two_columns) -> None:
            output = df_two_columns.physt.histogram()
            assert isinstance(output, Histogram2D)

        def test_single_column(self, df_one_column: pd.DataFrame) -> None:
            output = df_one_column.physt.histogram()
            assert isinstance(output, Histogram1D)


class TestH1ToSeries:
    def test_simple_h1(self, simple_h1: Histogram1D) -> None:
        output = simple_h1.to_series()  # type: ignore
        expected = pd.Series(
            [1, 25, 0, 12],
            index=IntervalIndex.from_breaks(
                [0, 1, 1.5, 2, 3], closed="left", dtype="interval[float64]", name="Name"
            ),
            name="frequency",
        )
        assert_series_equal(output, expected)

    # TODO: Add more tests?


class TestH1ToDataFrame:
    def test_simple_h1(self, simple_h1: Histogram1D) -> None:
        output = simple_h1.to_dataframe()  # type: ignore
        expected = pd.DataFrame(
            {"frequency": [1, 25, 0, 12], "error": [1, 5, 0, np.sqrt(12)]},
            index=IntervalIndex.from_breaks(
                [0, 1, 1.5, 2, 3], closed="left", dtype="interval[float64]", name="Name"
            ),
        )
        assert_frame_equal(output, expected)

    # TODO: Add more tests?
