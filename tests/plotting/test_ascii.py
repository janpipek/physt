from textwrap import dedent

from physt.plotting.ascii import hbar, map


def test_plot_hbar(simple_h1, capsys):
    hbar(simple_h1, show_values=True)
    captured = capsys.readouterr()
    assert captured.out == EXPECTED_HBAR


def test_plot_map(simple_h2, capsys):
    map(simple_h2)
    captured = capsys.readouterr()
    assert captured.out == EXPECTED_MAP


EXPECTED_HBAR = dedent(
    """██ 1
█████████████████████████████████████████████████████ 25
▏ 0
█████████████████████████ 12
"""
)


EXPECTED_MAP = dedent(
    """3.00 →
┌──┐
│██│6.00 ↑
│▒▓│
│ ░│4.00 ↓
└──┘
← 0.00
↓0
 ▒█
 6 ↑
"""
)
