from __future__ import absolute_import
from vispy import scene, io, visuals
from vispy.scene.visuals import Isosurface
from vispy.visuals import transforms
import numpy as np

types = ("voxel")

dims = {
    "voxel": [3]
}


def _get_canvas(kwargs):
    # TODO: add viewbox arg
    canvas = kwargs.pop("canvas", None)
    if canvas is None:
        canvas = scene.SceneCanvas(keys='interactive')
        canvas.size = 800, 600
        canvas.show()

    viewbox = scene.widgets.ViewBox(border_color='yellow', parent=canvas.scene)
    grid = canvas.central_widget.add_grid()
    grid.padding = 6
    grid.add_widget(viewbox, 0, 0)

    return canvas, viewbox


def isosurface(h3, level=None, **kwargs):
    if level is None:
        level = h3.frequencies.max() / 2
    canvas, viewbox = _get_canvas(kwargs)
    camera = scene.cameras.TurntableCamera(parent=viewbox.scene, fov=60)
    viewbox.camera = camera
    surface = Isosurface(h3.frequencies, level=level)



def voxel(h3, **kwargs):
    canvas, vb1 = _get_canvas(kwargs)
    grid = canvas.central_widget.add_grid()
    grid.padding = 6
    grid.add_widget(vb1, 0, 0)

    scenes = (vb1,)

    vol1 = h3.frequencies
    # volume1 = scene.visuals.Volume(vol1, parent=scenes, threshold=0.5)



    cam1 = scene.cameras.TurntableCamera(parent=vb1.scene, fov=60)
    vb1.camera = cam1

    box = scene.visuals.Cube(.51, color=(0, 0, 1, 0.2), parent=vb1.scene)
    box.transform = transforms
    box.transform.translate((1, 1, 1))

    scene.visuals.Polygon()

    box = np.array([[0, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [1, 0, 0],
                [0, 0, 0]], dtype=np.float32)
    box1 = scene.visuals.Line(pos=box, color=(0.7, 0, 0, 1), method='gl',
                          name='unit box', parent=vb1.scene)

    return canvas
