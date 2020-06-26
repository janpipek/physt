import sys
import os
sys.path = [os.path.join(os.path.dirname(__file__), "..")] + sys.path
import physt
import numpy as np
import pytest
from physt import special


class TestPolar:
    def test_simple_create(self):
        data = np.array([[0.01, 0.01], [0.01, 0.99], [-1, .01], [-1, -.01]])
        x = data[:,0]
        y = data[:,1]
        h = special.polar_histogram(x, y, radial_bins=2, phi_bins=4)
        assert h.bin_count == 8
        assert np.array_equal([[1, 0, 0, 0], [1, 1, 1, 0]], h.frequencies)

    def test_transform(self):
        t = special.PolarHistogram.transform([1, 0])
        assert np.array_equal(t, [1, 0])

        t = special.PolarHistogram.transform([0, 2])
        assert np.allclose(t, [2, np.pi / 2])

        data = np.asarray([[1, 0], [0, 2]])
        t = special.PolarHistogram.transform(data)
        assert np.allclose(t, [[1, 0], [2, np.pi / 2]])

    def test_projection_types(self):
        data = np.array([[0.01, 0.01], [0.01, 0.99], [-1, .01], [-1, -.01]])
        x = data[:, 0]
        y = data[:, 1]
        h = special.polar_histogram(x, y, radial_bins=2, phi_bins=4)
        assert special.RadialHistogram == type(h.projection("r"))
        assert special.AzimuthalHistogram == type(h.projection("phi"))


class TestSpherical:
    def test_transform(self):
        t = special.SphericalHistogram.transform([0, 0, 1])
        assert np.array_equal(t,  [1, 0, 0])

        t = special.SphericalHistogram.transform([2, 2, 0])
        assert np.array_equal(t,  [np.sqrt(8), np.pi / 2, np.pi / 4])

        data = np.asarray([[3, 0, 0], [0, 0, 0], [0, .5, -.5]])
        expected = np.asarray([[3, np.pi / 2, 0], [0, 0, 0], [np.sqrt(.5), .75 * np.pi, np.pi / 2]])
        assert np.allclose(expected, special.SphericalHistogram.transform(data))

    def test_projection_types(self):
        h = special.spherical_histogram([[1, 2, 3], [2, 3, 4]])
        assert special.SphericalSurfaceHistogram == type(h.projection("phi", "theta"))
        assert special.SphericalSurfaceHistogram == type(h.projection("theta", "phi"))

    def test_equal_radius(self):
        """Issue #62"""
        n = 1000
        data = np.empty((n, 3))
        np.random.seed(42)
        data[:,0] = np.random.normal(0, 1, n)
        data[:,1] = np.random.normal(0, 1, n)
        data[:,2] = np.random.normal(0, 1, n)
        for i in range(n):
            scale = np.sqrt(data[i,0] ** 2 + data[i,1] ** 2 + data[i,2] ** 2)
            data[i,0] = data[i,0] / scale
            data[i,1] = data[i,1] / scale
            data[i,2] = data[i,2] / scale

        with pytest.raises(ValueError) as exc:
            special.spherical_histogram(data, theta_bins=20, phi_bins=20)
        assert "All radii seem to be the same: 1.0000. Perhaps you wanted to use `spherical_surface_histogram` instead or set radius bins explicitly?" in str(exc)


class TestSphericalSurface:
    def test_simple_sphere_data(self):
        n = 100
        data = np.empty((n, 3))
        np.random.seed(42)
        data[:, 0] = np.random.normal(0, 1, n)
        data[:, 1] = np.random.normal(0, 1, n)
        data[:, 2] = np.random.normal(0, 1, n)

        h = special.spherical_surface_histogram(data, theta_bins=10, phi_bins=20)



class TestCylindricalSurface:
    def test_radius(self):
        h = special.cylindrical_histogram([[1,2,3], [2, 3, 4]])
        proj = h.projection("phi", "z")
        assert proj.radius > 1

    def test_projection_types(self):
        h = special.cylindrical_histogram([[1, 2, 3], [2, 3, 4]])
        proj = h.projection("phi", "z")
        assert special.AzimuthalHistogram == type(proj.projection("phi"))


class TestCylindrical:
    def test_transform(self):
        t = special.CylindricalHistogram.transform([0, 0, 1])
        assert np.array_equal(t,  [0, 0, 1])

        t = special.CylindricalHistogram.transform([2, 2, 2])
        assert np.array_equal(t,  [np.sqrt(8), np.pi / 4, 2])

        data = np.asarray([[3, 0, 0], [0, 0, 0], [0, .5, -.5]])
        expected = np.asarray([[3, 0, 0], [0, 0, 0], [.5, np.pi / 2, -.5]])
        assert np.allclose(expected, special.CylindricalHistogram.transform(data))

    def test_projection_types(self):
        h = special.cylindrical_histogram([[1, 2, 3], [2, 3, 4]])
        assert special.CylindricalSurfaceHistogram == type(h.projection("phi", "z"))
        assert special.CylindricalSurfaceHistogram == type(h.projection("z", "phi"))
        assert special.PolarHistogram == type(h.projection("rho", "phi"))
        assert special.PolarHistogram == type(h.projection("phi", "rho"))


if __name__ == "__main__":
    pytest.main(__file__)
