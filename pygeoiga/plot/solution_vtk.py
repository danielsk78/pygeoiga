import pyvista
import numpy as np


def p_temperature(x, y, temperature, p=None,show=False, **kwargs):
    if p is None:
        p = pyvista.Plotter(notebook=False)

    if "cmap" not in kwargs:
        kwargs["cmap"] = "viridis"

    #points = np.c_[x.reshape(-1), y.reshape(-1), temperature.reshape(-1)]
    #cloud = pyvista.PolyData(points)
    #surf = cloud.delaunay_2d()
    #surf['values'] = temperature.ravel()

    surf = pyvista.StructuredGrid(x, y, temperature)
    surf['values'] = temperature.T.ravel()
    p.add_mesh(surf, scalars="values", color="k", **kwargs)
    if show:
        from pygeoiga.plot.nrbplotting_vtk import p_show
        p_show(p)
    return p

