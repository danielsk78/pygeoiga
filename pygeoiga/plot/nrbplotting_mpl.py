import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#matplotlib.use('Qt5Agg')

def create_figure(typ="2d"):
    if typ=="2d":
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
    elif typ=="3d":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    return fig, ax

def p_cpoints(cp, dim = None, ax = None, point= True, line=True, show = False, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    if "color" not in kwargs:
        kwargs["color"] = "red"
    mrk = kwargs.pop("marker", "s")
    linst = kwargs.pop("linestyle","--")

    if dim == 3:
        if line:
            ax.plot_wireframe(cp[..., 0], cp[..., 1], cp[..., 2], linestyle=linst, **kwargs)
        if point:
            n, m = cp.shape[0], cp.shape[1]
            P = np.asarray([(cp[x, y, 0], cp[x, y, 1], cp[x, y, 2]) for y in range(m) for x in range(n)])
            ax.plot(P[:,0], P[:,1], P[:,2], marker=mrk, linestyle="None", **kwargs)
    elif dim == 2:
        X = cp[..., 0]
        Y = cp[..., 1]
        if line:
            if ax.name == "3d":
                ax.plot_wireframe(X, Y, np.zeros(cp.shape[:-1]), linestyle=linst, **kwargs)
            else:
                ax.plot(X, Y, linestyle=linst, **kwargs)
                ax.plot(X.T, Y.T, linestyle=linst, **kwargs)
        if point:
            #n, m = cp.shape[0], cp.shape[1]
            #P = np.asarray([(cp[x, y, 0], cp[x, y, 1]) for y in range(m) for x in range(n)])
            ax.plot(X, Y, marker=mrk, linestyle="None", **kwargs)
    elif dim == 1:
        if line:
            ax.plot(cp[:, 0], cp[:, 1], cp[:,2] if cp.shape[-1]>2 else np.zeros(cp.shape[0]), linestyle=linst, **kwargs)
        if point:
            ax.plot(cp[:, 0], cp[:, 1], cp[:,2] if cp.shape[-1]>2 else np.zeros(cp.shape[0]), marker=mrk, linestyle="None", **kwargs)
    if show:
        plt.show()
    return ax

def p_knots(knots, cp, ax = None, dim = 3, point= True, line=True, show = False, resolution =25, **kwargs):
    from pygeoiga.engine.NURB_engine import curve_point, surface_point
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    if isinstance(knots, list) or isinstance(knots, tuple):
        knots = np.asarray(knots)
    if "color" not in kwargs:
        kwargs["color"] = "green"
    mkr = kwargs.pop("marker", "^")
    linst = kwargs.pop("linestyle", "-")

    if knots.shape[0] == 1:
        knots=knots[0]
        degree = len(np.where(knots == 0.)[0]) - 1
        weight = np.ones(cp.shape[0])
        points = np.linspace(knots[0], knots[-1], resolution)
        if line:
            pos = []
            for u in points:
                pos.append(curve_point(u, degree, knots, cp, weight))
            pos = np.asarray(pos)
            if dim == 3:
                ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], linestyle=linst, **kwargs)
            if dim == 2:
                ax.plot(pos[:, 0], pos[:, 1], linestyle=linst, **kwargs)
        if point:
            pos = []
            for u in knots:
                pos.append(curve_point(u, degree, knots, cp, weight))
            pos = np.asarray(pos)
            if dim == 3:
                ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                         linestyle="None", marker=mkr, **kwargs)
            if dim == 2:
                ax.plot(pos[:, 0], pos[:, 1],
                        linestyle="None", marker=mkr, **kwargs)

    else:
        degree1 = len(np.where(np.asarray(knots[0]) == 0.)[0]) - 1
        degree2 = len(np.where(np.asarray(knots[1]) == 0.)[0]) - 1
        weight = np.ones((cp.shape[0], cp.shape[1], 1))
        knot1 = np.asarray(knots[0])
        knot2 = np.asarray(knots[1])

        points1 = np.linspace(knot1[0], knot1[-1], resolution)
        points2 = np.linspace(knot2[0], knot2[-1], resolution)

        if line:
            for u in knot1:
                positions_eta = []
                for v in points1:
                    positions_eta.append(surface_point(u,
                                              v,
                                              degree1,
                                              degree2,
                                              knot1,
                                              knot2,
                                              cp,
                                              weight))
                positions_eta = np.asarray(positions_eta)
                if dim == 3:
                    ax.plot(positions_eta[:,0], positions_eta[:,1], positions_eta[:,2], linestyle=linst, **kwargs)
                if dim == 2:
                    ax.plot(positions_eta[:, 0], positions_eta[:, 1], linestyle=linst, **kwargs)

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
                if dim == 3:
                    ax.plot(positions_eta[:,0], positions_eta[:,1], positions_eta[:,2], linestyle=linst, **kwargs)
                if dim == 2:
                    ax.plot(positions_eta[:, 0], positions_eta[:, 1], linestyle=linst, **kwargs)

        if point:

            positions_eta = []
            for u in knot1:
                for v in knot2:
                    positions_eta.append(surface_point(u,
                                              v,
                                              degree1,
                                              degree2,
                                              knot1,
                                              knot2,
                                              cp,
                                              weight))
            positions_eta = np.asarray(positions_eta)
            if dim == 3:
                ax.plot(positions_eta[:,0], positions_eta[:,1], positions_eta[:,2], linestyle="None", marker=mkr, **kwargs)
            if dim == 2:
                ax.plot(positions_eta[:, 0], positions_eta[:, 1], linestyle="None", marker=mkr, **kwargs)
    if show:
        plt.show()

    return ax

def p_curve(knots=None, cp=None, ax = None, weight= None, positions = None,  dim=3, show = False, resolution =30, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    if "color" not in kwargs:
        kwargs["color"] = "blue"
    if positions is None:
        from pygeoiga.engine.NURB_engine import NURBS_Curve
        positions = np.asarray(NURBS_Curve(cp, knots, resolution, weight)).T
    if dim == 3:
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], **kwargs)
    if dim == 2:
        ax.plot(positions[:, 0], positions[:, 1], **kwargs)
    if show:
        plt.show()
    return ax

def p_surface(knots=None,
              cp=None,
              weight = None,
              positions = None,
              dim = 3,
              ax = None,
              resolution=20,
              border=True,
              fill=True,
              show = False,
              **kwargs):

    if ax is None:
        if dim==3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        elif dim==2:
            fig,ax =plt.subplots()
    if "color" not in kwargs:
        kwargs["color"] = "blue"

    if positions is None:
        from pygeoiga.engine.NURB_engine import NURBS_Surface
        positions = NURBS_Surface(cp, knots, resolution, weight)

    if dim == 3:
        ax.plot_trisurf(positions[:, 0], positions[:, 1], positions[:, 2], **kwargs)
    if dim == 2:
        from pygeoiga.analysis.common import position_borders_nrb
        pos = position_borders_nrb(cp, knots, resolution)
        if isinstance(ax, Axes3D):
            ax.add_collection3d(Poly3DCollection([list(zip(pos[:,0],pos[:,1], np.zeros(len(pos))))], **kwargs))
            #ax.plot_trisurf(positions[:, 0], positions[:, 1], np.zeros(positions.shape[0]), **kwargs)
        else:
            if border:
                ax.plot(pos[:,0], pos[:,1], **kwargs)
            if fill:
                ax.fill(pos[:,0], pos[:,1], **kwargs)
    if show:
        plt.show()
    return ax

def plot_volume():
    raise NotImplementedError