from threading import Thread

import pytest
import numpy as np

from physt import h1

class TestH1:
    @staticmethod
    def _threaded_fill(h, data, n_threads) -> None:
        def _fill():
            for d in data:
                h.fill(d)

        threads = [Thread(target=_fill) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    @pytest.mark.parametrize(
        "n_threads", [1, 5]
    )
    def test_h1(self, n_threads: int):
        h = h1(data=None, bins=[0, 1, 2, 3, 4])
        data = np.linspace(0, 5, 1000)
        self._threaded_fill(h, data, n_threads)
        assert h.total == 800 * n_threads

    @pytest.mark.parametrize(
        "n_threads", [100]
    )
    def test_adaptive_h1(self, n_threads: int):
        h = h1(data=None, bins="fixed_width", bin_width=1, adaptive=True)
        n_points = 100
        data = np.linspace(0, 100, n_points)
        self._threaded_fill(h, data, n_threads)
        assert h.total == n_points * n_threads
