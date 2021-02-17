import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d.axes3d import Axes3D


def p_temperature_mp(geometry, ax, inter_points=100, **kwargs):

    x_temp = np.asarray([])
    y_temp = np.asarray([])
    t_temp = np.asarray([])
    for patch_id in geometry.keys():
        x_temp = np.hstack([x_temp, geometry[patch_id]["x_sol"].ravel()])
        y_temp = np.hstack([y_temp, geometry[patch_id]["y_sol"].ravel()])
        t_temp = np.hstack([t_temp, geometry[patch_id]["t_sol"].ravel()])
    # Set up a regular grid of interpolation points
    #xi = np.linspace(x_temp.min(), x_temp.max(), inter_points)
    #yi = np.linspace(y_temp.min(), y_temp.max(), inter_points)
    #xi, yi = np.meshgrid(xi, yi)

    #import scipy.interpolate
    #ti = scipy.interpolate.griddata((x_temp, y_temp), t_temp, (xi, yi), method='linear')

    ax=p_temperature(x_pos = x_temp, y_pos = y_temp, temperature=t_temp, ax=ax,**kwargs)
                  #extent=[x_temp.min(), x_temp.max(), y_temp.min(), y_temp.max()],  **kwargs)
    return ax
cbar = None
def p_temperature(x_pos, y_pos, temperature,
                  vmin = None,
                  vmax = None,
                  ax = None,
                  show = False,
                  colorbar = True,
                  point = False,
                  fill=True,
                  contour=False,
                  dim=2,
                  **kwargs):
    """

    Args:
        x_pos:
        y_pos:
        temperature:
        ax:
        show:
        colorbar:
        **kwargs:

    Returns:

    """
    if vmin is not None:
        kwargs["vmin"] = vmin
    else:
        vmin = np.min(temperature)
        kwargs["vmin"] = vmin
    if vmax is not None:
        kwargs["vmax"] = vmax
    else:
        vmax = np.max(temperature)
        kwargs["vmax"] = vmax
    con = None
    sca = None
    color = kwargs.pop("color", None)
    colors = kwargs.pop("colors", "black") # contour lines
    cmap = kwargs.pop("cmap", "viridis")
    lev = kwargs.pop("levels", 10)
    if isinstance(lev, int):
        levels = np.linspace(vmin, vmax, lev)
    else:
        levels=lev
    extent = kwargs.pop("extent", None)
    if ax is None:
        if dim==3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        elif dim==2:
            fig, ax = plt.subplots()
    if point:
        #ax.plot(x_pos.ravel(), y_pos.ravel(), color = color, marker=".", linestyle="None")
        sca = ax.scatter(x_pos, y_pos, c=temperature if color is None else color, marker="o", vmin=vmin, vmax=vmax, zorder=100)
    if contour:
        if x_pos.ndim == 2 and y_pos.ndim == 2:
            con = ax.contour(x_pos, y_pos, temperature, levels=levels, zorder=50, colors=colors, **kwargs)
        elif x_pos.ndim == 1 and y_pos.ndim == 1:
            con = ax.tricontour(x_pos, y_pos, temperature, levels=levels, zorder=50, colors=colors, **kwargs)
        else:
            raise
        #con = ax.contour(x_pos, y_pos, temperature, levels=levels, zorder=50, colors=colors, **kwargs)
        ax.clabel(con, fmt="%.1f")
    if fill:
        if isinstance(ax, Axes3D):
            if x_pos is not None and y_pos is not None:
                if x_pos.ndim == 2 and y_pos.ndim ==2:
                    con = ax.plot_surface(x_pos, y_pos, temperature, cmap=cmap, **kwargs)
                elif x_pos.ndim == 1 and y_pos.ndim ==1:
                    tri = mtri.Triangulation(x_pos, y_pos)
                    con = ax.plot_trisurf(tri, temperature, cmap=cmap, **kwargs)
                else:
                    raise
            else:
                con = ax.plot_surface(temperature, extent=extent, cmap=cmap, **kwargs)
        else:
            #con = ax.contourf(x_pos,y_pos,temperature, cmap=cmap, levels=levels, zorder=-1, **kwargs)
            if x_pos is not None and y_pos is not None:
                if x_pos.ndim == 2 and y_pos.ndim ==2:
                    con = ax.contourf(x_pos, y_pos, temperature, cmap=cmap, levels=levels, zorder=-1, **kwargs)
                elif x_pos.ndim == 1 and y_pos.ndim ==1:
                    con = ax.tricontourf(x_pos, y_pos, temperature, cmap=cmap, levels=levels, zorder=-1, **kwargs)
                else:
                    raise
                #con = ax.pcolormesh(x_pos, y_pos, temperature, cmap=cmap, zorder=-1,shading='gouraud', **kwargs)
            else:
                con = ax.contourf(temperature, cmap=cmap, levels=levels, zorder=-1, extent=extent, **kwargs)
                #con = ax.pcolormesh(temperature, cmap=cmap, zorder=-1, shading='gouraud', **kwargs)

    global cbar
    if colorbar: # TODO: if using subplots this does not work anymore
        if len(ax.figure.axes) == 1 or cbar is None: #check if already have a colorbar
            if isinstance(ax, Axes3D):
                cbar = plt.colorbar(con if con is not None else sca, label="Temperature")
            else:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad="2%")
                norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
                mappeable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
                cbar = ax.figure.colorbar(mappeable, cax=cax, ax=ax, label="Temperature",
                                          format="%.1f")
    else:
        cbar = None
    if show:
        plt.show()

    return ax

def p_triangular_mesh(x, y, ax, dim=2, **kwargs):
    if ax is None:
        if dim==3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        elif dim==2:
            fig, ax = plt.subplots()

    triang = mtri.Triangulation(x, y)

    if isinstance(ax, Axes3D):
        ax.plot_trisurf(triang, **kwargs)
    else:
        ax.triplot(triang, **kwargs)

    return ax

