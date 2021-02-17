import pyvista
import numpy as np

def create_figure():
    p = pyvista.Plotter(notebook=False)
    return p

def p_cpoints(cp, p=None, show = False, **kwargs):
    if p is None:
        p = pyvista.Plotter(notebook=False)
    if "color" not in kwargs:
        kwargs["color"] = "red"
    if "point_size" not in kwargs:
        kwargs["point_size"] = 15
    n, m = cp.shape[0], cp.shape[1]
    P = np.asarray([(cp[x, y, 0], cp[x, y, 1], cp[x, y, 2] if cp.shape[-1]==3 else 0) for y in range(m) for x in range(n)])
    sp = pyvista.PolyData(P)
    p.add_mesh(sp,  **kwargs)
    if show:
        p_show(p)
    return p

def p_knots(knots, cp, p = None, show = False, resolution = 20, point=True, line=True, **kwargs):
    from pygeoiga.engine.NURB_engine import curve_point, surface_point

    if p is None:
        p = pyvista.Plotter(notebook=False)
    if isinstance(knots, list):
        knots = np.asarray(knots)
    if cp.shape[-1]==2:
        new_cp = np.zeros((cp.shape[0], cp.shape[1], 3))
        new_cp[..., :cp.shape[-1]] = cp
        cp = new_cp
    if "color" not in kwargs:
        kwargs["color"] = "green"

    point_size = kwargs.pop("point_size", 15)
    line_width = kwargs.pop("line_width", 4)

    if knots.shape[0] == 1:
        degree = len(np.where(knots == 0.)[0]) - 1
        weight = np.ones(cp.shape[0])
        if line:
            points = np.linspace(knots[0], knots[-1], resolution)
            pos = []
            for u in points:
                pos.append(curve_point(u, degree, knots, cp, weight))
            pos = np.asarray(pos)
            spline = pyvista.Spline(pos, resolution**2)
            p.add_mesh(spline, line_width=line_width, **kwargs)
        if point:
            pos = []
            for u in knots:
                pos.append(curve_point(u, degree, knots, cp, weight))
            pos = np.asarray(pos)
            ptn = pyvista.PolyData(pos)
            p.add_mesh(ptn, point_size=point_size, **kwargs)
    else:
        degree1 = len(np.where(knots[0] == 0.)[0]) - 1
        degree2 = len(np.where(knots[1] == 0.)[0]) - 1
        weight = np.ones((cp.shape[0], cp.shape[1], 1))
        knot1 = np.asarray(knots[0])
        knot2 = np.asarray(knots[1])

        points1 = np.linspace(knot1[0], knot1[-1], resolution)
        points2 = np.linspace(knot2[0], knot2[-1], resolution)
        if line:
            for u in knot1:
                positions_xi = []
                for v in points1:
                    positions_xi.append(surface_point(u,
                                                       v,
                                                       degree1,
                                                       degree2,
                                                       knot1,
                                                       knot2,
                                                       cp,
                                                       weight))
                positions_xi = np.asarray(positions_xi)
                spline_xi = pyvista.Spline(positions_xi, resolution ** 2)
                p.add_mesh(spline_xi, line_width=line_width, **kwargs)

            for v in knot2:
                positions_eta = []
                for u in points2:
                    positions_eta.append(surface_point(u,
                                                       v,
                                                       degree1,
                                                       degree2,
                                                       knot1,
                                                       knot2,
                                                       cp,
                                                       weight))
                positions_eta = np.asarray(positions_eta)
                spline_eta = pyvista.Spline(positions_eta, resolution ** 2)
                p.add_mesh(spline_eta, line_width=line_width, **kwargs)

        if point:
            positions=[]
            for u in knot1:
                for v in knot2:
                    positions.append(surface_point(u,
                                                      v,
                                                      degree1,
                                                      degree2,
                                                      knot1,
                                                      knot2,
                                                      cp,
                                                      weight))
            positions = np.asarray(positions)
            ptn = pyvista.PolyData(positions)
            p.add_mesh(ptn, point_size=point_size, **kwargs)
    if show:
        p_show(p)

    return p

def p_surface(knots, cp, interactive=True, weight = None, positions = None, p = None, resolution=20, show = False, **kwargs):
    from pygeoiga.engine.NURB_engine import NURBS_Surface
    if p is None:
        p = pyvista.Plotter(notebook=False)
    color = kwargs.pop("color", "blue")
    radius = kwargs.pop("radius", 5.0)
    opacity = kwargs.pop("opacity", 1.0)
    if cp.shape[-1]==2:
        new_cp = np.zeros((cp.shape[0], cp.shape[1], 3))
        new_cp[..., :cp.shape[-1]] = cp
        cp = new_cp
    if positions is None:
        positions = NURBS_Surface(cp, knots, resolution, weight)
    cloud = pyvista.PolyData(positions)
    surf = cloud.delaunay_2d()
    p.add_mesh(surf, color=color, opacity=opacity, **kwargs)

    if interactive:
        n, m = cp.shape[0], cp.shape[1]
        P = np.asarray([(cp[x, y, 0], cp[x, y, 1], cp[x, y, 2]) for y in range(m) for x in range(n)])

        def update_surface(point, i):
            P[i] = point
            temp = P
            cp_new = np.reshape(temp.ravel(), cp.shape)
            positions = NURBS_Surface(cp_new, knots, resolution, weight)
            cloud.points = positions
            return

        p.add_sphere_widget(update_surface, center=P, radius=radius, **kwargs )

    if show:
        p_show(p)
    return p

def p_show(p):
    #p.set_background("white")
    p.show_axes()
    p.show_grid(color="black")
    p.show(cpos="xy")
