from textwrap import dedent
import pytest
from physt.histogram1d import Histogram1D
from physt.plotting.ascii import hbar, map


@pytest.fixture
def simple_h1():
    return Histogram1D(
        binning=[0, 1, 2, 3, 4],
        frequencies=[1, 2, 7, 0],
    )


def test_plot_hbar(simple_h1, capsys):
    hbar(simple_h1, show_values=True, show_labels=True, label_width=4, max_width=29)
    captured = capsys.readouterr()
    assert captured.out == EXPECTED_HBAR


def test_plot_map(simple_h2, capsys):
    map(simple_h2)
    captured = capsys.readouterr()
    assert captured.out == EXPECTED_MAP


EXPECTED_HBAR = """   0 ███ 1
   1 ██████ 2
   2 ██████████████████████ 7
   3 ▏ 0
   4 
"""


EXPECTED_MAP = dedent(
    """\
3.00 →
┌───┐
│░▓█│6.00 ↑
│ ▒█│4.00 ↓
└───┘
← 0.00
↓0
 ▒█
 6 ↑
"""
)
