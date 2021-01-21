import numpy as np
import pytest

from physt import special_histograms
from physt.special_histograms import AzimuthalHistogram, azimuthal

@pytest.fixture
def empty_azimuthal() -> AzimuthalHistogram:
    return azimuthal(np.zeros(0,), np.zeros(0,))


class TestAzimuthal:
    def test_simple_create(self):
        data = np.array([[0.01, 0.01], [0.01, 0.99], [-1, .01], [-1, -.01]])
        x = data[:,0]
        y = data[:,1]
        h = special_histograms.azimuthal(x, y, bins=4)
        assert h.axis_name == "phi"
        assert h.bin_count == 4
        assert np.array_equal([2, 1, 1, 0], h.frequencies)

    def test_transform(self):
        t = AzimuthalHistogram.transform([1, 0])
        assert np.array_equal(t, 0)

        t = AzimuthalHistogram.transform([0, 2])
        assert np.allclose(t, np.pi / 2)

        data = np.asarray([[1, 0], [0, 2]])
        t = AzimuthalHistogram.transform(data)
        assert np.allclose(t, [0, np.pi / 2])

    def test_correct_find_bin(self, empty_azimuthal):
        assert empty_azimuthal.find_bin(1, transformed=True) == 2
        assert empty_azimuthal.find_bin((0.5, 0.877)) == 2

    def test_incorrect_find_bin(self, empty_azimuthal):
        with pytest.raises(ValueError) as exc:
            empty_azimuthal.find_bin(1)
        assert exc.match("AzimuthalHistogram can transform only")
        with pytest.raises(ValueError) as exc:
            empty_azimuthal.find_bin((1, 2), transformed=True)
        assert exc.match("Non-scalar value for 1D histogram")


class TestPolar:
    def test_simple_create(self):
        data = np.array([[0.01, 0.01], [0.01, 0.99], [-1, .01], [-1, -.01]])
        x = data[:,0]
        y = data[:,1]
        h = special_histograms.polar(x, y, radial_bins=2, phi_bins=4)
        assert h.axis_names == ("r", "phi")
        assert h.bin_count == 8
        assert np.array_equal([[1, 0, 0, 0], [1, 1, 1, 0]], h.frequencies)

    def test_transform(self):
        t = special_histograms.PolarHistogram.transform([1, 0])
        assert np.array_equal(t, [1, 0])

        t = special_histograms.PolarHistogram.transform([0, 2])
        assert np.allclose(t, [2, np.pi / 2])

        data = np.asarray([[1, 0], [0, 2]])
        t = special_histograms.PolarHistogram.transform(data)
        assert np.allclose(t, [[1, 0], [2, np.pi / 2]])

    def test_densities(self):
        h = special_histograms.PolarHistogram(
            binnings=[[0, 1, 2], [0, 1, 2]],
            frequencies=[[1, 2], [3, 4]]
        )
        assert np.array_equal(h.densities, [[2, 4], [2, 4 / 1.5]])

    def test_projection_types(self):
        data = np.array([[0.01, 0.01], [0.01, 0.99], [-1, .01], [-1, -.01]])
        x = data[:, 0]
        y = data[:, 1]
        h = special_histograms.polar(x, y, radial_bins=2, phi_bins=4)
        assert special_histograms.RadialHistogram == type(h.projection("r"))
        assert special_histograms.AzimuthalHistogram == type(h.projection("phi"))


class TestRadial:
    def test_simple_create(self):
        data = np.array([[0.01, 0.01, 1], [0.01, 0.99, 1], [-1, .01, 1], [-1, -.01, 1]])
        x = data[:,0]
        y = data[:,1]
        z = data[:,2]
        h = special_histograms.radial(x, y)
        assert h.axis_name == "r"

        h_xyz = special_histograms.radial(x, y, z)
        h_3d = special_histograms.radial(data)

        assert h_xyz == h_3d


    def test_transform(self):
        t = special_histograms.RadialHistogram.transform([1, 0])
        assert t == 1

        t = special_histograms.RadialHistogram.transform([1, 1, 1])
        assert np.allclose(t, np.sqrt(3))

        with pytest.raises(ValueError) as exc:
            special_histograms.RadialHistogram.transform([1, 1, 1, 1])


class TestSpherical:
    def test_simple_create(self):
        pass

    def test_transform(self):
        t = special_histograms.SphericalHistogram.transform([0, 0, 1])
        assert np.array_equal(t,  [1, 0, 0])

        t = special_histograms.SphericalHistogram.transform([2, 2, 0])
        assert np.array_equal(t,  [np.sqrt(8), np.pi / 2, np.pi / 4])

        data = np.asarray([[3, 0, 0], [0, 0, 0], [0, .5, -.5]])
        expected = np.asarray([[3, np.pi / 2, 0], [0, 0, 0], [np.sqrt(.5), .75 * np.pi, np.pi / 2]])
        assert np.allclose(expected, special_histograms.SphericalHistogram.transform(data))

    def test_projection_types(self):
        h = special_histograms.spherical([[1, 2, 3], [2, 3, 4]])
        assert special_histograms.SphericalSurfaceHistogram == type(h.projection("phi", "theta"))
        assert special_histograms.SphericalSurfaceHistogram == type(h.projection("theta", "phi"))

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
            special_histograms.spherical(data, theta_bins=20, phi_bins=20)
        assert exc.match("All radii seem to be the same")


class TestSphericalSurface:
    def test_simple_sphere_data(self):
        n = 100
        data = np.empty((n, 3))
        np.random.seed(42)
        data[:, 0] = np.random.normal(0, 1, n)
        data[:, 1] = np.random.normal(0, 1, n)
        data[:, 2] = np.random.normal(0, 1, n)

        h = special_histograms.spherical_surface(data, theta_bins=10, phi_bins=20)


class TestCylindricalSurface:
    def test_radius(self):
        h = special_histograms.cylindrical([[1, 2, 3], [2, 3, 4]])
        proj = h.projection("phi", "z")
        assert proj.radius > 1

    def test_projection_types(self):
        h = special_histograms.cylindrical([[1, 2, 3], [2, 3, 4]])
        proj = h.projection("phi", "z")
        assert special_histograms.AzimuthalHistogram == type(proj.projection("phi"))


class TestCylindrical:
    def test_transform(self):
        t = special_histograms.CylindricalHistogram.transform([0, 0, 1])
        assert np.array_equal(t,  [0, 0, 1])

        t = special_histograms.CylindricalHistogram.transform([2, 2, 2])
        assert np.array_equal(t,  [np.sqrt(8), np.pi / 4, 2])

        data = np.asarray([[3, 0, 0], [0, 0, 0], [0, .5, -.5]])
        expected = np.asarray([[3, 0, 0], [0, 0, 0], [.5, np.pi / 2, -.5]])
        assert np.allclose(expected, special_histograms.CylindricalHistogram.transform(data))

    def test_projection_types(self):
        h = special_histograms.cylindrical([[1, 2, 3], [2, 3, 4]])
        assert special_histograms.CylindricalSurfaceHistogram == type(h.projection("phi", "z"))
        assert special_histograms.CylindricalSurfaceHistogram == type(h.projection("z", "phi"))
        assert special_histograms.PolarHistogram == type(h.projection("rho", "phi"))
        assert special_histograms.PolarHistogram == type(h.projection("phi", "rho"))
