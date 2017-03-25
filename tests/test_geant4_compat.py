import sys
import os
sys.path = [os.path.join(os.path.dirname(__file__), "..")] + sys.path
from physt import h1, h2, histogramdd
from physt.compat import geant4
import numpy as np
import pytest


class TestGeant4Compat(object):
    def test_read_h1(self):
        path = os.path.join(os.path.dirname(__file__), "data/geant-h1.csv")
        h = geant4.load_csv(path)
        assert h.shape == (100,)
        assert h.ndim == 1
        assert h.name == "Edep in absorber"
        assert h.total == 10000

    def test_read_h2(self):
        path = os.path.join(os.path.dirname(__file__), "data/geant-h2.csv")
        h = geant4.load_csv(path)
        assert h.ndim == 2
        assert h.shape == (50,50)
        assert h.name == "Drift Chamber 1 X vs Y"
        assert h.total == 292


if __name__ == "__main__":
    pytest.main(__file__)
