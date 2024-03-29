import numpy as np
import matplotlib.pyplot as plt
# analysis
from pygeoiga.analysis.MultiPatch import patch_topology, form_k_IGA_mp, boundary_condition_mp, map_MP_elements
from pygeoiga.analysis.common import solve
# Plotting
from pygeoiga.plot.nrbplotting_mpl import p_surface, p_cpoints, p_knots
from pygeoiga.plot.solution_mpl import p_temperature, p_triangular_mesh, p_temperature_mp
# Exporting to Fenics and/or Moose
from pygeoiga.nurb.nrb_to_gmsh import convert_geometry_mp_to_gmsh
from pygeoiga.FE_solvers.run_fenics import convert_msh_to_xdmf, run_simulation


import pygeoiga as gn
import os
datapath = os.path.abspath(gn.myPath+"/../tests/chapter_figures/data/") + os.sep
fig_folder=gn.myPath+'/../../manuscript_IGA_MasterThesis/Thesis/figures/07_Examples/'
kwargs_savefig=dict(transparent=True, box_inches='tight', pad_inches=0)
save_all=False

def plot_IGA(geometry, a, gDoF, figsize=(5,5), file_name="temp.pdf", save=False, levels=None):
    figsize=(figsize[0]*2, figsize[1])
    fig_sol, [ax2, ax3] = plt.subplots(1, 2, figsize=figsize, sharey=True)

    xmin=0
    xmax=0
    ymin=0
    ymax=0
    cbar=True
    for patch_id in geometry.keys():
        x = geometry[patch_id].get("x_sol")
        y = geometry[patch_id].get("y_sol")
        t = geometry[patch_id].get("t_sol")

        xmin = x.min() if x.min() < xmin else xmin
        xmax = x.max() if x.max() > xmax else xmax
        ymin = y.min() if y.min() < ymin else ymin
        ymax = y.max() if y.max() > ymax else ymax

        ax2 = p_temperature(x, y, t, vmin=np.min(a), vmax=np.max(a), show=False, colorbar=False,
                            ax=ax2, point=True, fill=False,
                            markersize=25)

        ax2 = p_knots(geometry[patch_id].get("knots"),
                      geometry[patch_id].get("B"),
                      ax=ax2,
                      color='k',
                      dim=2,
                      point=False,
                      line=True,
                      linestyle="--",
                      linewidth=0.2)

        ax3 = p_temperature(x, y, t, vmin=np.min(a), vmax=np.max(a), levels=200, show=False, colorbar=cbar, ax=ax3,
                            point=False, fill=True, contour=False)
        cbar = False

    ax2.set_title("%s DoF"%gDoF)

    ax2.set_ylabel(r"$y$")
    for ax in ax2, ax3:
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$")
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)

    for c in ax3.collections:
        c.set_edgecolor("face")

    for patch_id in geometry.keys():
        x = geometry[patch_id].get("x_sol")
        y = geometry[patch_id].get("y_sol")
        t = geometry[patch_id].get("t_sol")

        ax3 = p_temperature(x, y, t, vmin=np.min(a), vmax=np.max(a), levels=levels,
                            show=False, colorbar=False, colors=["black"], ax=ax3,
                            point=False, fill=False, contour=True, cmap=None,
                            linewidths=0.5)
    #ax3 = p_temperature_mp(geometry, vmin=np.min(a), vmax=np.max(a), levels=levels,
    #                                         show=False, colorbar=False, colors=["black"], ax=ax3,
    #                                            point=False, fill=False, contour=True, cmap=None,
    #                                            linewidths=0.5)

    plt.tight_layout()
    fig_sol.show()

    if save or save_all:
        fig_sol.savefig(fig_folder + file_name, **kwargs_savefig)

def plot_fenics(nodal_coordinates, temperature_nodes, figsize=(5,5), file_name="fencis_temp.pdf", save=False, levels=None,
                geometry=None, u=None):
    figsize = (figsize[0]*2, figsize[1])
    fig_sol, [ax2, ax3] = plt.subplots(1, 2, figsize=figsize, sharey=True)

    ax2 = p_triangular_mesh(nodal_coordinates[:,0],
                            nodal_coordinates[:,1],
                            color = "black",
                            ax=ax2,
                            linestyle="--",
                            linewidth=0.2
                            )

    ax2 = p_temperature(nodal_coordinates[:, 0],
                        nodal_coordinates[:, 1],
                        temperature_nodes,
                        vmin=temperature_nodes.min(),
                        vmax=temperature_nodes.max(),
                        ax=ax2,
                        point=True,
                        fill=False,
                        contour=False,
                        colorbar=False,
                        markersize=25
                        )

    if u is not None and geometry is not None:
        for patch_id in geometry.keys():
            x = geometry[patch_id].get("x_sol")
            y = geometry[patch_id].get("y_sol")
            m, n = x.shape
            t = np.zeros((m, n))
            for x_i in range(n):
                for y_i in range(m):
                    t[y_i, x_i] = u(x[y_i, x_i], y[y_i, x_i], 0)

            ax3 = p_temperature(x,
                                y,
                                t,
                                vmin=temperature_nodes.min(),
                                vmax=temperature_nodes.max(),
                                ax=ax3,
                                point=False,
                                fill=True,
                                contour=False,
                                colorbar=True,
                                levels=100)

    else:
        ax3 = p_temperature(nodal_coordinates[:, 0],
                            nodal_coordinates[:, 1],
                            temperature_nodes,
                            vmin=temperature_nodes.min(),
                            vmax=temperature_nodes.max(),
                            ax=ax3,
                            point=False,
                            fill=True,
                            contour=False,
                            colorbar=True,
                            levels=100)


    for c in ax3.collections:
        c.set_edgecolor("face")

    ax3 = p_temperature(nodal_coordinates[:, 0],
                        nodal_coordinates[:, 1],
                        temperature_nodes,
                        vmin=temperature_nodes.min(),
                        vmax=temperature_nodes.max(),
                        ax=ax3,
                        point=False,
                        fill=False,
                        contour=True,
                        colorbar=False,
                        cmap=None,
                        levels=levels,
                        linewidths=0.5)

    ax2.set_ylabel(r"$y$")
    for ax in ax2, ax3:
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$")
        ax.set_xlim(nodal_coordinates[:, 0].min(),nodal_coordinates[:, 0].max())
        ax.set_ylim(nodal_coordinates[:, 1].min(),nodal_coordinates[:, 1].max())

    ax2.set_title("%s DoF"%len(temperature_nodes))
    plt.tight_layout()
    fig_sol.show()

    if save or save_all:
        fig_sol.savefig(fig_folder + file_name, **kwargs_savefig)

def workflow(geometry_callable, knot_ins, size, name, T_t, T_b, T_l, T_r, levels, save=False, **kwargs):
    figsize= kwargs.pop("figsize", (5, 5))
    jump=kwargs.pop("jump",1)
    lith = kwargs.pop("lith", None)
    start = kwargs.pop("start", 0)
    original = kwargs.pop("original", False)

    geometry = geometry_callable(refine=True, knot_ins=knot_ins)
    geometry, gDoF = patch_topology(geometry)
    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_IGA_mp(geometry, K_glob)

    a = np.zeros(gDoF)
    F = np.zeros(gDoF)

    bc, a = boundary_condition_mp(geometry, a, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    a, F = solve(bc, K_glob, F, a)
    geometry = map_MP_elements(geometry, a)
    plot_IGA(geometry, a, gDoF,figsize, name + ".pdf", save, levels=levels, )

    geometry2 = geometry_callable(refine=True)
    mesh, script, physical_tag_id = convert_geometry_mp_to_gmsh(geometry2,
                                                                size=size,
                                                                save_geo=datapath + name + ".geo",
                                                                save_msh=datapath + name + ".msh")
    input = datapath + name + ".msh"
    convert_msh_to_xdmf(input)
    nodal_coordinates, temperature_nodes, mesh, u = run_simulation(input,
                                                                   topology_info=physical_tag_id,
                                                                   top_bc=T_t,
                                                                   bot_bc=T_b,
                                                                   geometry=geometry2,
                                                                   show=False)

    plot_fenics(nodal_coordinates, temperature_nodes,figsize, name + "_fenics.pdf", save, levels=levels,
                geometry=geometry, u=u)
    if original:
        fig, ax = plt.subplots(figsize=figsize)
        for patch_id in geometry.keys():
            ax = p_surface(geometry[patch_id].get("knots"),
                           geometry[patch_id].get("B"),
                           color=geometry[patch_id].get("color"),
                           dim=2,
                           fill=True,
                           border=True,
                           ax=ax,
                           alpha=0.7)

        ax.set_aspect("equal")
        ax.set_ylabel(r"$y$")
        ax.set_xlabel(r"$x$")
        ax.set_xlim(nodal_coordinates[:, 0].min(), nodal_coordinates[:, 0].max())
        ax.set_ylim(nodal_coordinates[:, 1].min(), nodal_coordinates[:, 1].max())
        if lith is None:
            lith = list(geometry.keys())[start::jump]
        ax.legend(labels=lith,
                  handles=ax.patches[start::jump],
                  loc='upper left',
                  bbox_to_anchor=(0.05, .9),
                  borderaxespad=0)
        plt.tight_layout()
        fig.show()

        if save or save_all:
            fig.savefig(fig_folder + name+"_geometry.pdf", **kwargs_savefig)

def do_IGA(function, T_t, T_b, knot_ins):
    geometry =  function(refine=True, knot_ins=knot_ins)
    from pygeoiga.analysis.MultiPatch import patch_topology, form_k_IGA_mp
    geometry, gDoF = patch_topology(geometry)
    K_glob_IGA = np.zeros((gDoF, gDoF))
    F_IGA = np.zeros(gDoF)
    a_IGA = np.zeros(gDoF)
    K_glob_IGA = form_k_IGA_mp(geometry, K_glob_IGA)
    from pygeoiga.analysis.MultiPatch import boundary_condition_mp
    bc_IGA, a_IGA = boundary_condition_mp(geometry, a_IGA, T_t, T_b, None, None)
    bc_IGA["gDOF"] = gDoF
    from pygeoiga.analysis.common import solve
    a_IGA, F_IGA = solve(bc_IGA, K_glob_IGA, F_IGA, a_IGA)
    from pygeoiga.analysis.MultiPatch import map_MP_elements
    geometry = map_MP_elements(geometry, a_IGA)
    return geometry, gDoF

def do_Bezier(function, T_t, T_b, knot_ins):
    bezier_geometry = function(refine=True, knot_ins=knot_ins)
    from pygeoiga.analysis.MultiPatch import patch_topology, bezier_extraction_mp, form_k_IGA_mp, form_k_bezier_mp
    bezier_geometry, gDoF = patch_topology(bezier_geometry)
    bezier_geometry = bezier_extraction_mp(bezier_geometry)
    K_glob_be = np.zeros((gDoF, gDoF))
    F_be = np.zeros(gDoF)
    a_be = np.zeros(gDoF)
    K_glob_be = form_k_bezier_mp(bezier_geometry, K_glob_be)
    from pygeoiga.analysis.MultiPatch import boundary_condition_mp
    bc_be, a_be = boundary_condition_mp(bezier_geometry, a_be, T_t, T_b, None, None)
    bc_be["gDOF"] = gDoF
    from pygeoiga.analysis.common import solve
    a_be, F_be = solve(bc_be, K_glob_be, F_be, a_be)
    from pygeoiga.analysis.MultiPatch import map_MP_elements
    bezier_geometry = map_MP_elements(bezier_geometry, a_be)
    return bezier_geometry, gDoF

def do_FEM(function, T_t, T_b, knot_ins , size):
    geometry = function(refine=True, knot_ins=knot_ins)
    name = "temporal"
    mesh, script, physical_tag_id = convert_geometry_mp_to_gmsh(geometry,
                                                                size=size,
                                                                save_geo=datapath + name + ".geo",
                                                                save_msh=datapath + name + ".msh")
    input = datapath + name + ".msh"
    convert_msh_to_xdmf(input)
    nodal_coordinates, temperature_nodes, mesh, u = run_simulation(input,
                                                                   topology_info=physical_tag_id,
                                                                   top_bc=T_t,
                                                                   bot_bc=T_b,
                                                                   geometry=geometry,
                                                                   show=False)
    return nodal_coordinates, temperature_nodes, u


def same_IGA_FEM(geometry, T_t, T_b, filepath):
    from pygeoiga.analysis.MultiPatch import patch_topology, bezier_extraction_mp, form_k_IGA_mp, form_k_bezier_mp
    geometry, gDoF = patch_topology(geometry)

    K_glob_IGA = np.zeros((gDoF, gDoF))
    F_IGA = np.zeros(gDoF)
    a_IGA = np.zeros(gDoF)

    K_glob_IGA = form_k_IGA_mp(geometry, K_glob_IGA)

    from pygeoiga.analysis.MultiPatch import boundary_condition_mp
    bc_IGA, a_IGA = boundary_condition_mp(geometry, a_IGA, T_t, T_b, None, None)
    bc_IGA["gDOF"] = gDoF

    from pygeoiga.analysis.common import solve

    a_IGA, F_IGA = solve(bc_IGA, K_glob_IGA, F_IGA, a_IGA)

    from pygeoiga.analysis.MultiPatch import map_MP_elements

    geometry = map_MP_elements(geometry, a_IGA)

    plot_IGA(geometry, a_IGA, gDoF)

    from pygeoiga.FE_solvers.run_fenics import read_fenics_solution
    u, mesh, dofs = read_fenics_solution(filepath)
    u.set_allow_extrapolation(True) # TODO: Not understand when this is needed
    fig, ax = plt.subplots()
    min_p = None
    max_p = None
    cmap = plt.get_cmap("seismic")
    save_sol = dict()
    for patch_id in geometry.keys():
        x = geometry[patch_id].get("x_sol")
        y = geometry[patch_id].get("y_sol")
        m, n = x.shape
        correct = np.zeros((m, n))

        for x_i in range(n):
            for y_i in range(m):
                correct[y_i,x_i] = u(x[y_i,x_i], y[y_i,x_i],  0)
        err = geometry[patch_id].get("t_sol") - correct
        save_sol[patch_id] = err
        if min_p is None or min_p > err.min():
            min_p = err.min()
        if max_p is None or max_p < err.max():
            max_p = err.max()

    #if np.abs(min_p) > np.abs(max_p):
    #    max_p = np.abs(min_p)
    #else:
    #    min_p = -np.abs(max_p)

    for patch_id in geometry.keys():
        x = geometry[patch_id].get("x_sol")
        y = geometry[patch_id].get("y_sol")
        err = save_sol[patch_id]
        ax.contourf(x, y, err, vmin=min_p, vmax=max_p, cmap=cmap)

    import matplotlib
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="2%")
    norm = matplotlib.colors.TwoSlopeNorm(vmin=min_p, vcenter=0, vmax=max_p)
    #norm = matplotlib.colors.Normalize(vmin=min_p, vmax=max_p, v)
    mappeable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = ax.figure.colorbar(mappeable, cax=cax, ax=ax, label="Temperature [°C]")

    fig.show()

def comparison_all_meshes(function_callable, T_t, T_b, filepath, size=100,
                          knot_ins=[np.arange(0.1,1,0.1), np.arange(0.1,1,0.1)],
                          bezier=True,
                          save=False,
                          name="temp.none",
                          label="Difference [°C] (Solution - "):
    import matplotlib
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    geometry_IGA, dof_IGA = do_IGA(function_callable, T_t, T_b, knot_ins)

    geometry_BE, dof_BE = do_Bezier(function_callable, T_t, T_b, knot_ins)
    coor, temp, u_FEM = do_FEM(function_callable, T_t, T_b, knot_ins, size)
    u_FEM.set_allow_extrapolation(True)
    dof_FEM = len(temp)

    from pygeoiga.FE_solvers.run_fenics import read_fenics_solution
    u, mesh, dofs = read_fenics_solution(filepath)
    u.set_allow_extrapolation(True)  # TODO: Not understand when this is needed

    if bezier:
        fig, [ax_IGA, ax_BE, ax_FEM] = plt.subplots(1,3, sharey=True, figsize=(17,5))
    else:
        fig, [ax_IGA, ax_FEM] = plt.subplots(1, 2, sharey=True, figsize=(11, 5))
    #cma = plt.cm.seismic
    #cma = plt.cm.Reds
    cma = plt.cm.RdBu
    def max_min(IGA, BE, FEM):
        diff_all = np.array([])
        for geometry in [IGA, BE]:
            for patch_id in geometry.keys():
                x = geometry[patch_id].get("x_sol")
                y = geometry[patch_id].get("y_sol")
                m, n = x.shape
                correct = np.zeros((m, n))
                for x_i in range(n):
                    for y_i in range(m):
                        correct[y_i, x_i] = u(x[y_i, x_i], y[y_i, x_i], 0)

                err = correct - geometry[patch_id].get("t_sol")
                #err = (correct - geometry[patch_id].get("t_sol"))**2
                diff_all = np.r_[diff_all, err.ravel()]

        t_fun = np.vectorize(u)
        val = t_fun(coor[:, 0], coor[:, 1], np.zeros(dof_FEM))
        erro = val - temp
        diff_all = np.r_[diff_all, erro.ravel()]

        vmin = np.min(diff_all)
        vmax = np.max(diff_all)
        #if vmin == 0:
        #    vmin = -1e-3
        #if vmax==0:
        #    vmax = 1e-3
        return vmin, vmax

    def geometry_difference(geometry, u, ax, vmin, vmax):
        x_all = np.array([])
        y_all = np.array([])
        diff_all = np.array([])
        count = 0
        for patch_id in geometry.keys():
            x = geometry[patch_id].get("x_sol")
            y = geometry[patch_id].get("y_sol")
            m, n = x.shape
            correct = np.zeros((m, n))
            for x_i in range(n):
                for y_i in range(m):
                    correct[y_i, x_i] = u(x[y_i, x_i], y[y_i, x_i], 0)

            err = correct - geometry[patch_id].get("t_sol")
            #err = (correct - geometry[patch_id].get("t_sol")) ** 2

            x_all = np.r_[x_all, x.ravel()]
            y_all = np.r_[y_all, y.ravel()]
            diff_all = np.r_[diff_all, err.ravel()]
            count +=1

        vmin = np.min(diff_all)
        vmax = np.max(diff_all)

        #if np.abs(vmin)<vmax:
        #    vmin=-vmax
        #else:
        #    vmax = np.abs(vmin)

        if vmin >= 0:
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            val = cma(np.linspace(0.5, 1, 256))
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list('seismic_2', val)
        elif vmax <= 0:
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            val = cma(np.linspace(0, 0.5, 256))
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list('seismic_2', val)
        else:
            norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            cmap = cma
        #cmap = cma
        mappeable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

        if count == 1:
            ax.contourf(x, y, err, vmin=vmin, vmax=vmax, cmap=cmap, levels=100)
        else:
            ax.tricontourf(x_all, y_all, diff_all, vmin=vmin, vmax=vmax, cmap=cmap, levels=100)

        divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad="2%")
        cax = divider.append_axes("bottom", size="5%", pad="10%")
        cbar = ax.figure.colorbar(mappeable, cax=cax, ax=ax, label=label+"IGA)", orientation="horizontal", format='%.0e')
        return ax
    vmin, vmax = max_min(geometry_IGA, geometry_BE, u)
    ax_IGA = geometry_difference(geometry_IGA, u, ax_IGA, vmin, vmax)
    ax_IGA.set_title("%s DoF IGA" %dof_IGA)
    if bezier:
        ax_BE = geometry_difference(geometry_BE, u, ax_BE,  vmin, vmax)
        ax_BE.set_title("%s DoF Bezier" %dof_BE)

    #FEM
    count=0
    x_all = np.array([])
    y_all = np.array([])
    diff_all = np.array([])
    for patch_id in geometry_IGA.keys():
        x = geometry_IGA[patch_id].get("x_sol")
        y = geometry_IGA[patch_id].get("y_sol")
        m, n = x.shape
        correct = np.zeros((m, n))
        correct_FEM = np.zeros((m, n))
        for x_i in range(n):
            for y_i in range(m):
                correct_FEM[y_i, x_i] = u_FEM(x[y_i, x_i], y[y_i, x_i], 0)
                correct[y_i, x_i] = u(x[y_i, x_i], y[y_i, x_i], 0)
        erro = correct - correct_FEM
        #erro = (correct-correct_FEM)**2
        diff_all = np.r_[diff_all, erro.ravel()]
        x_all = np.r_[x_all, x.ravel()]
        y_all = np.r_[y_all, y.ravel()]
        count +=1
    vmin=np.min(diff_all)
    vmax=np.max(diff_all)
    #if np.abs(vmin) < vmax:
    #    vmin = -vmax
    #else:
    #    vmax = np.abs(vmin)
    #t_fun = np.vectorize(u)
    #val = t_fun(coor[:,0], coor[:,1], np.zeros(dof_FEM))
    #erro = (val - temp)**2
    #vmin = np.min(erro)
    #vmax = np.max(erro)
    #count=2
    if vmin == 0:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        val = cma(np.linspace(0.5, 1, 256))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('seismic_2', val)

    elif vmax == 0:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        val = cma(np.linspace(0, 0.5, 256))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('seismic_2', val)
    else:
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        cmap = cma
    #cmap = cma
    # norm = matplotlib.colors.Normalize(vmin=min_p, vmax=max_p, v)
    if count == 1:
        ax_FEM.contourf(x, y, erro, vmin=vmin, vmax=vmax, cmap=cmap, levels=100)
    else:
        ax_FEM.tricontourf(x_all, y_all, diff_all, vmin=vmin, vmax=vmax, cmap=cmap, levels=100)

    #ax_FEM.tricontourf(coor[:, 0], coor[:, 1], erro, vmin=vmin, vmax=vmax, cmap=cmap, levels=100)
    ax_FEM.set_title("%s DoF FEM" % dof_FEM)

    divider = make_axes_locatable(ax_FEM)
    cax = divider.append_axes("bottom", size="5%", pad="10%")

    mappeable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = ax_FEM.figure.colorbar(mappeable, cax=cax, ax=ax_FEM, label=label+"FEM)", orientation="horizontal", format='%.0e')

    for c in ax_IGA.collections:
        c.set_edgecolor("face")

    for c in ax_FEM.collections:
        c.set_edgecolor("face")
    fig.show()

    if save:
        fig.savefig(fig_folder+name+".pdf", **kwargs_savefig)

def test_plot_solution_3_layer_mp():
    from pygeoiga.nurb.cad import make_3_layer_patches
    save = False
    levels=[11, 12, 14, 17, 20, 22,23, 24]
    T_t = 10
    T_b = 25
    T_l = None
    T_r = None
    name = "3_layer_anticline"
    lith = ["Granite: 3.1 [W/mK]", "Mudstone: 0.9 [W/mK]", "Sandstone: 3 [W/mK]"]

    workflow(make_3_layer_patches,
             knot_ins=([0.25, 0.5, 0.75],[0.25, 0.5, 0.75]),
             size=65,
             name=name,
             T_t=T_t,
             T_b=T_b,
             T_l=T_l,
             T_r=T_r,
             save=save,
             levels=levels,
             lith=lith,
             original=False)
    #comparison_all_meshes(make_3_layer_patches,
    #                      T_t=T_t,
    #                      T_b=T_b,
    #                      filepath=datapath + "solution_anticline",
    #                      size=80,
    #                      knot_ins=([0.25, 0.5, 0.75],[0.25, 0.5, 0.75]),
    #                      bezier=False,
    #                      save=save,
    #                      name=name + "_error")
    workflow(make_3_layer_patches,
             knot_ins=(np.arange(0.1,1,0.1),np.arange(0.1,1,0.1)),
             size=28,
             name=name+"_refined",
             T_t=T_t,
             T_b=T_b,
             T_l=T_l,
             T_r=T_r,
             save=save,
             levels=levels,
             original=False)
    #comparison_all_meshes(make_3_layer_patches,
    #                      T_t=T_t,
    #                      T_b=T_b,
    #                      filepath=datapath + "solution_anticline",
    #                      size=40,
    #                      knot_ins=([0.1,0.2,0.3, 0.4,0.5, 0.6,0.7, 0.8,0.9],[0.25, 0.5, 0.75]),
    #                      bezier=False,
    #                      save=save,
    #                      name=name + "_refined_error")

def test_plot_solution_fault_model_mp():
    from pygeoiga.nurb.cad import make_fault_model
    save = False
    levels = [12, 16, 20, 24, 28, 32, 36]
    T_t = 10
    T_b = 40
    T_l = None
    T_r = None
    name = "fault_model"
    lith = ["Sandstone: 3.1 [W/mK]", "Mudstone: 0.9 [W/mK]", "Sandstone: 3 [W/mK]"]

    workflow(make_fault_model,
             knot_ins=([1/3,2/3], [1/3,2/3]),
             size=95,
             name=name,
             T_t=T_t,
             T_b=T_b,
             T_l=T_l,
             T_r=T_r,
             save=save,
             levels=levels,
             jump=2,
             start=4,
             lith = lith,
             original=False)
    workflow(make_fault_model,
             knot_ins=(np.arange(0.15,1,0.15), np.arange(0.15,1,0.15)),
             size=45,
             name=name + "_refined",
             T_t=T_t,
             T_b=T_b,
             T_l=T_l,
             T_r=T_r,
             save=save,
             levels=levels,
             original=False)

def test_plot_solution_salt_dome_mp():
    from pygeoiga.nurb.cad import make_salt_dome
    save = False
    levels = [15, 20, 30, 40, 50, 60, 70, 80, 85]
    T_t = 10
    T_b = 90
    T_l = None
    T_r = None
    name = "salt_dome"
    lith = ["Granite: 3.1 [W/mK]",
            "Salt: 7.5 [W/mK]",
            "Shale: 1.2 [W/mK]",
            "Sandstone: 3 [Wm/K]",
            "Claystone: 0.9 [Wm/K]",
            "Sandstone: 3.2 [W/mK]"]

    workflow(make_salt_dome,
             figsize=(7,4),
             jump=3,
             knot_ins=[[0.3, 0.65, 0.8], [0.3, 0.65, 0.8]],
             size=228,
             name=name,
             T_t=T_t,
             T_b=T_b,
             T_l=T_l,
             T_r=T_r,
             save=save,
             levels=levels,
             original=False,
             lith=lith)
    workflow(make_salt_dome,
             figsize=(7,4),
             jump=3,
             knot_ins=([0.1, 0.3, 0.4, 0.6, 0.8, 0.9], [0.1, 0.3, 0.4, 0.6, 0.8, 0.9]),
             size=135,
             name=name + "_refined",
             T_t=T_t,
             T_b=T_b,
             T_l=T_l,
             T_r=T_r,
             save=save,
             levels=levels)

def test_plot_solution_3_layer_mp_mesh():
    from pygeoiga.nurb.cad import make_3_layer_patches
    geometry = make_3_layer_patches(refine=True, knot_ins=([0.5], [0.5]))

    fig, [ax1, ax2] = plt.subplots(1,2, figsize=(10,5), sharey=True)
    for patch_id in geometry.keys():
        ax1 = p_knots(geometry[patch_id].get("knots"),
                      geometry[patch_id].get("B"),
                      ax=ax1,
                      color='k',
                      dim=2,
                      point=False,
                      line=True,
                      linestyle="-",
                      linewidth=1)
    name = "mesh_anticline"
    mesh, script, physical_tag_id = convert_geometry_mp_to_gmsh(geometry,
                                                                size=200,
                                                                save_geo=datapath + name + ".geo",
                                                                save_msh=datapath + name + ".msh")
    input = datapath + name + ".msh"
    convert_msh_to_xdmf(input)
    nodal_coordinates, temperature_nodes, mesh, u = run_simulation(input,
                                                                   topology_info=physical_tag_id,
                                                                   top_bc=10,
                                                                   bot_bc=40,
                                                                   geometry=geometry,
                                                                   show=False)

    ax2 = p_triangular_mesh(nodal_coordinates[:, 0],
                            nodal_coordinates[:, 1],
                            color="black",
                            ax=ax2,
                            linestyle="-",
                            linewidth=0.2
                            )

    for ax in ax1, ax2:
        ax.set_aspect("equal")
        ax.set_ylabel(r"$y$")
        ax.set_xlabel(r"$x$")
        ax.set_xlim(nodal_coordinates[:, 0].min(),nodal_coordinates[:, 0].max())
        ax.set_ylim(nodal_coordinates[:, 1].min(),nodal_coordinates[:, 1].max())

    plt.tight_layout()
    fig.show()

def same_IGA_BEZIER(geometry, T_t, T_b):
    from pygeoiga.analysis.MultiPatch import patch_topology, bezier_extraction_mp, form_k_IGA_mp, form_k_bezier_mp
    geometry, gDoF = patch_topology(geometry)

    import copy
    bezier_geometry = copy.deepcopy(geometry)
    bezier_geometry = bezier_extraction_mp(bezier_geometry)

    K_glob_IGA = np.zeros((gDoF, gDoF))
    F_IGA = np.zeros(gDoF)
    a_IGA = np.zeros(gDoF)
    K_glob_be = np.zeros((gDoF, gDoF))
    F_be = np.zeros(gDoF)
    a_be = np.zeros(gDoF)

    K_glob_IGA = form_k_IGA_mp(geometry, K_glob_IGA)
    K_glob_be = form_k_bezier_mp(bezier_geometry, K_glob_be)

    from pygeoiga.analysis.MultiPatch import boundary_condition_mp
    bc_IGA, a_IGA = boundary_condition_mp(geometry, a_IGA, T_t, T_b, None, None)
    bc_IGA["gDOF"] = gDoF
    bc_be, a_be = boundary_condition_mp(bezier_geometry, a_be, T_t, T_b, None, None)
    bc_be["gDOF"] = gDoF
    from pygeoiga.analysis.common import solve

    a_IGA, F_IGA = solve(bc_IGA, K_glob_IGA, F_IGA, a_IGA)
    a_be, F_be = solve(bc_be, K_glob_be, F_be, a_be)

    from pygeoiga.analysis.MultiPatch import map_MP_elements

    geometry = map_MP_elements(geometry, a_IGA)
    bezier_geometry = map_MP_elements(bezier_geometry, a_be)

    plot_IGA(geometry, a_IGA, gDoF)

    plot_IGA(bezier_geometry, a_IGA, gDoF)

    fig, ax = plt.subplots()
    min_p = None
    max_p = None
    cmap = plt.get_cmap("seismic")
    for patch_id in geometry.keys():
        err = geometry[patch_id].get("t_sol") - bezier_geometry[patch_id].get("t_sol")
        x = geometry[patch_id].get("x_sol")
        y = geometry[patch_id].get("y_sol")
        if min_p is None or min_p > err.min():
            min_p = err.min()
        if max_p is None or max_p < err.max():
            max_p = err.max()

    #if np.abs(min_p) > np.abs(max_p):
    #    max_p = np.abs(min_p)
    #else:
    #    min_p = -np.abs(max_p)

    for patch_id in geometry.keys():
        err = geometry[patch_id].get("t_sol") - bezier_geometry[patch_id].get("t_sol")
        x = geometry[patch_id].get("x_sol")
        y = geometry[patch_id].get("y_sol")
        ax.contourf(x, y, err, vmin=min_p, vmax=max_p, cmap=cmap)

    import matplotlib
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="2%")
    norm = matplotlib.colors.TwoSlopeNorm(vmin=min_p, vcenter=0, vmax=max_p)
    #norm = matplotlib.colors.Normalize(vmin=min_p, vmax=max_p, v)
    mappeable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = ax.figure.colorbar(mappeable, cax=cax, ax=ax, label="Temperature")

    fig.show()

def test_same_anticline():
    from pygeoiga.nurb.cad import make_3_layer_patches
    geometry = make_3_layer_patches(refine=True)
    same_IGA_BEZIER(geometry, 10, 25)

def test_same_fault():
    from pygeoiga.nurb.cad import make_fault_model
    geometry = make_fault_model(refine=True)
    same_IGA_BEZIER(geometry, 10, 40)

def test_same_dome():
    from pygeoiga.nurb.cad import make_salt_dome
    geometry = make_salt_dome(refine=False)
    knot_ins = [np.arange(0.1, 1, 0.1), np.arange(0.1, 1, 0.1)]
    for patch_id in geometry.keys():
        knots = geometry[patch_id].get("knots")
        knot_ins[0] = [x for x in knot_ins[0] if x not in knots[0]]
        knot_ins[1] = [x for x in knot_ins[1] if x not in knots[1]]

    geometry = make_salt_dome(refine=True, knot_ins=knot_ins)
    same_IGA_BEZIER(geometry, 10, 90)

def create_high_resolution_answers(geometry_callable, T_t, T_b, name, size):
    geometry = geometry_callable(refine=True)
    mesh, script, physical_tag_id = convert_geometry_mp_to_gmsh(geometry,
                                                                size=size,
                                                                save_geo=datapath + name + ".geo",
                                                                save_msh=datapath + name + ".msh")
    input = datapath + name + ".msh"
    convert_msh_to_xdmf(input)
    nodal_coordinates, temperature_nodes, mesh, u = run_simulation(input,
                                                                   topology_info=physical_tag_id,
                                                                   top_bc=T_t,
                                                                   bot_bc=T_b,
                                                                   geometry=geometry,
                                                                   show=False,
                                                                   save_solution=True)

def test_save_3_layer_solution():
    from pygeoiga.nurb.cad import make_3_layer_patches
    size = 1
    name = "solution_anticline"
    T_t = 10
    T_b = 25
    create = True
    if create:
        create_high_resolution_answers(make_3_layer_patches,T_t, T_b, name, size)

    from pygeoiga.FE_solvers.run_fenics import read_fenics_solution
    u, mesh, dofs = read_fenics_solution(datapath + name + ".h5")
    print(u(10,10,0), dofs)  # dofs = 1159004 # current 290771

def test_save_fault_solution():
    from pygeoiga.nurb.cad import make_fault_model
    size = 1
    name = "solution_fault"
    T_t = 10
    T_b = 40
    create = False
    if create:
        create_high_resolution_answers(make_fault_model, T_t, T_b, name, size)

    from pygeoiga.FE_solvers.run_fenics import read_fenics_solution
    u, mesh, dofs = read_fenics_solution(datapath + name + ".h5")
    print(u(10,10, 0), dofs)  # dofs = 1160679

def test_save_salt_dome_solution():
    from pygeoiga.nurb.cad import make_salt_dome
    size = 4
    name = "solution_salt_dome"
    T_t = 10
    T_b = 90
    create = True
    if create:
        create_high_resolution_answers(make_salt_dome, T_t, T_b, name, size)

    from pygeoiga.FE_solvers.run_fenics import read_fenics_solution
    u, mesh, dofs = read_fenics_solution(datapath + name + ".h5")
    print(u(10,10, 0), dofs) # 1308251

def test_same_anticline_IGA_FEM():
    from pygeoiga.nurb.cad import make_3_layer_patches
    geometry = make_3_layer_patches(refine=True)
    filepath = datapath + "solution_anticline.h5"
    same_IGA_FEM(geometry, 10, 25, filepath=filepath)

def test_same_fault_IGA_FEM():
    from pygeoiga.nurb.cad import make_fault_model
    geometry = make_fault_model(refine=True)
    filepath = datapath + "solution_fault.h5"
    same_IGA_FEM(geometry, 10, 40, filepath=filepath)

def test_same_fault_IGA_FEM():
    from pygeoiga.nurb.cad import make_salt_dome
    geometry = make_salt_dome(refine=True)
    filepath = datapath + "solution_salt_dome.h5"
    same_IGA_FEM(geometry, 10, 90, filepath=filepath)

def test_compare_all_meshes_anticline():
    from pygeoiga.nurb.cad import make_3_layer_patches

    comparison_all_meshes(make_3_layer_patches,
                          T_t=10,
                          T_b=25,
                          filepath = datapath + "solution_anticline.h5",
                          size=30,
                          knot_ins=[np.arange(0.1,1,0.1), np.arange(0.1,1,0.1)])

def test_compare_all_meshes_anticline_fine():
    from pygeoiga.nurb.cad import make_3_layer_patches

    comparison_all_meshes(make_3_layer_patches,
                          T_t=10,
                          T_b=25,
                          filepath = datapath + "solution_anticline.h5",
                          size=5,
                          knot_ins=[np.arange(0.02,1,0.02), np.arange(0.02,1,0.02)])


def test_compare_all_meshes_fault():
    from pygeoiga.nurb.cad import make_fault_model

    comparison_all_meshes(make_fault_model,
                          T_t=10,
                          T_b=40,
                          filepath = datapath + "solution_fault.h5",
                          size=30,
                          knot_ins=[np.arange(0.1,1,0.1), np.arange(0.1,1,0.1)])

def test_compare_all_meshes_fault_fine():
    from pygeoiga.nurb.cad import make_fault_model

    comparison_all_meshes(make_fault_model,
                          T_t=10,
                          T_b=40,
                          filepath = datapath + "solution_fault.h5",
                          size=20,
                          knot_ins=[np.arange(0.05,1,0.05), np.arange(0.05,1,0.05)])

def test_compare_all_meshes_dome():
    from pygeoiga.nurb.cad import make_salt_dome

    comparison_all_meshes(make_salt_dome,
                          T_t=10,
                          T_b=90,
                          filepath = datapath + "solution_salt_dome.h5",
                          size=200,
                          knot_ins=[np.arange(0.2,1,0.2), np.arange(0.2,1,0.2)])

def test_compare_all_meshes_dome_fine():
    from pygeoiga.nurb.cad import make_salt_dome

    comparison_all_meshes(make_salt_dome,
                          T_t=10,
                          T_b=90,
                          filepath = datapath + "solution_salt_dome.h5",
                          size=60,
                          knot_ins=[np.arange(0.1,1,0.1), np.arange(0.1,1,0.1)])


def test_compare_biquadratic():
    from pygeoiga.nurb.cad import make_surface_biquadratic
    knots, B = make_surface_biquadratic()
    from pygeoiga.nurb.refinement import knot_insertion
    knot_ins = [np.arange(0.1,1,0.1), np.arange(0.1,1,0.1)]

    knots_ins_0 = [x for x in knot_ins[0] if x not in knots[0]]
    B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_0, direction=0)
    knots_ins_1 = [x for x in knot_ins[1] if x not in knots[1]]
    B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_1, direction=1)

    geometry=dict()
    geometry["quadrat"] = {"B": B,
                            "knots": knots,
                            "kappa": 4,
                            'color': "red",
                            "position": (1,1),
                            "BC": {0: "bot_bc", 2: "top_bc"}
                           }
    T_t = 10
    T_b = 20
    geometry, gDoF = patch_topology(geometry)
    K_glob_IGA = np.zeros((gDoF, gDoF))
    F_IGA = np.zeros(gDoF)
    a_IGA = np.zeros(gDoF)
    K_glob_IGA = form_k_IGA_mp(geometry, K_glob_IGA)
    from pygeoiga.analysis.MultiPatch import boundary_condition_mp
    bc_IGA, a_IGA = boundary_condition_mp(geometry, a_IGA, T_t, T_b, None, None)
    bc_IGA["gDOF"] = gDoF
    from pygeoiga.analysis.common import solve
    a_IGA, F_IGA = solve(bc_IGA, K_glob_IGA, F_IGA, a_IGA)
    from pygeoiga.analysis.MultiPatch import map_MP_elements
    geometry = map_MP_elements(geometry, a_IGA)

    from pygeoiga.plot.nrbplotting_mpl import p_knots, create_figure
    fig, ax = create_figure()
    for patch_id in geometry.keys():
        ax = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"),
                     color=geometry[patch_id].get("color"), ax=ax, dim=2, point=False, line=True)
        x = geometry[patch_id].get("x_sol")
        y = geometry[patch_id].get("y_sol")
        t = geometry[patch_id].get("t_sol")
        ax = p_temperature(x, y, t, vmin=np.min(a_IGA), vmax=np.max(a_IGA), levels=50, show=False, colorbar=True, ax=ax,
                           point=True, fill=True, color="k")

    plt.show()

def test_biquadratic_bezier():
    from pygeoiga.nurb.cad import make_surface_biquadratic
    knots, B = make_surface_biquadratic()
    from pygeoiga.nurb.refinement import knot_insertion
    knot_ins = [np.arange(0.1, 1, 0.1), np.arange(0.1, 1, 0.1)]

    knots_ins_0 = [x for x in knot_ins[0] if x not in knots[0]]
    B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_0, direction=0)
    knots_ins_1 = [x for x in knot_ins[1] if x not in knots[1]]
    B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_1, direction=1)

    geometry = dict()
    geometry["quadrat"] = {"B": B,
                           "knots": knots,
                           "kappa": 4,
                           'color': "red",
                           "position": (1, 1),
                           "BC": {0: "bot_bc", 2: "top_bc"}}
    T_t = 10
    T_b = 20
    from pygeoiga.analysis.MultiPatch import patch_topology, bezier_extraction_mp, form_k_IGA_mp, form_k_bezier_mp
    geometry, gDoF = patch_topology(geometry)
    geometry = bezier_extraction_mp(geometry)
    K_glob_be = np.zeros((gDoF, gDoF))
    F_be = np.zeros(gDoF)
    a_be = np.zeros(gDoF)
    K_glob_be = form_k_bezier_mp(geometry, K_glob_be)
    from pygeoiga.analysis.MultiPatch import boundary_condition_mp
    bc_be, a_be = boundary_condition_mp(geometry, a_be, T_t, T_b, None, None)
    bc_be["gDOF"] = gDoF
    from pygeoiga.analysis.common import solve
    a_be, F_be = solve(bc_be, K_glob_be, F_be, a_be)
    from pygeoiga.analysis.MultiPatch import map_MP_elements
    geometry = map_MP_elements(geometry, a_be)

    from pygeoiga.plot.nrbplotting_mpl import p_knots, create_figure
    fig, ax = create_figure()
    for patch_id in geometry.keys():
        ax = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"),
                     color=geometry[patch_id].get("color"), ax=ax, dim=2, point=False, line=True)
        x = geometry[patch_id].get("x_sol")
        y = geometry[patch_id].get("y_sol")
        t = geometry[patch_id].get("t_sol")
        ax = p_temperature(x, y, t, vmin=np.min(a_be), vmax=np.max(a_be), levels=50, show=False, colorbar=True, ax=ax,
                           point=True, fill=True, color="k")

    plt.show()

def test_biquadratic_FEM():

    from pygeoiga.nurb.cad import make_surface_biquadratic
    knots, B = make_surface_biquadratic()
    from pygeoiga.nurb.refinement import knot_insertion
    knot_ins = [np.arange(0.1, 1, 0.1), np.arange(0.1, 1, 0.1)]

    knots_ins_0 = knot_ins[0]
    B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_0, direction=0)
    knots_ins_1 = knot_ins[1]
    B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_1, direction=1)

    geometry = dict()
    geometry["quadrat"] = {"B": B,
                           "knots": knots,
                           "kappa": 4,
                           'color': "red",
                           "position": (1, 1),
                           "BC": {0: "bot_bc", 2: "top_bc"}}
    T_t = 10
    T_b = 20
    size = 0.1
    name = "temporal"
    mesh, script, physical_tag_id = convert_geometry_mp_to_gmsh(geometry,
                                                                size=size,
                                                                save_geo=datapath + name + ".geo",
                                                                save_msh=datapath + name + ".msh")
    input = datapath + name + ".msh"
    convert_msh_to_xdmf(input)
    nodal_coordinates, temperature_nodes, mesh, u = run_simulation(input,
                                                                   topology_info=physical_tag_id,
                                                                   top_bc=T_t,
                                                                   bot_bc=T_b,
                                                                   geometry=geometry,
                                                                   show=True)


def test_save_biquadratic_high():
    def create_geom(**kwargs):
        from pygeoiga.nurb.cad import make_surface_biquadratic
        knots, B = make_surface_biquadratic()
        from pygeoiga.nurb.refinement import knot_insertion
        to_ins = np.arange(0.1, 1, 0.1)

        knots_ins_0 = [x for x in to_ins if x not in knots[0]]
        B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_0, direction=0)
        knots_ins_1 = [x for x in to_ins if x not in knots[1]]
        B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_1, direction=1)

        geometry = dict()
        geometry["quadrat"] = {"B": B,
                               "knots": knots,
                               "kappa": 4,
                               'color': "red",
                               "position": (1, 1),
                               "BC": {0: "bot_bc", 2: "top_bc"}}
        return geometry
    size = 0.005
    name = "solution_biquadratic"
    T_t = 10
    T_b = 25
    create = False

    if create:
        create_high_resolution_answers(create_geom, T_t, T_b, name, size)

    from pygeoiga.FE_solvers.run_fenics import read_fenics_solution
    u, mesh, dofs = read_fenics_solution(datapath + name + ".h5")
    print(u(0,2,0), dofs)  # dofs = 713526

def test_compare_all_meshes_biquadratic():
    def create_geom(**kwargs):
        from pygeoiga.nurb.cad import make_surface_biquadratic
        knots, B = make_surface_biquadratic()
        from pygeoiga.nurb.refinement import knot_insertion
        to_ins = np.arange(0.1, 1, 0.1)

        knots_ins_0 = [x for x in to_ins if x not in knots[0]]
        B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_0, direction=0)
        knots_ins_1 = [x for x in to_ins if x not in knots[1]]
        B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_1, direction=1)

        geometry = dict()
        geometry["quadrat"] = {"B": B,
                               "knots": knots,
                               "kappa": 4,
                               'color': "red",
                               "position": (1, 1),
                               "BC": {0: "bot_bc", 2: "top_bc"}}
        return geometry

    size = 0.4
    comparison_all_meshes(create_geom,
                          T_t=10,
                          T_b=25,
                          filepath = datapath + "solution_biquadratic.h5",
                          size=size,
                          knot_ins=[np.arange(0.1,1,0.1), np.arange(0.1,1,0.1)])

def test_compare_all_meshes_biquadratic_fine():
    def create_geom(**kwargs):
        from pygeoiga.nurb.cad import make_surface_biquadratic
        knots, B = make_surface_biquadratic()
        from pygeoiga.nurb.refinement import knot_insertion

        to_ins = np.arange(0.03, 1, 0.03)

        knots_ins_0 = [x for x in to_ins if x not in knots[0]]
        B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_0, direction=0)
        knots_ins_1 = [x for x in to_ins if x not in knots[1]]
        B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_1, direction=1)

        geometry = dict()
        geometry["quadrat"] = {"B": B,
                               "knots": knots,
                               "kappa": 4,
                               'color': "red",
                               "position": (1, 1),
                               "BC": {0: "bot_bc", 2: "top_bc"}}
        return geometry

    size = 0.12
    comparison_all_meshes(create_geom,
                          T_t=10,
                          T_b=25,
                          filepath = datapath + "solution_biquadratic.h5",
                          size=size,
                          knot_ins=[np.arange(0.1,1,0.1), np.arange(0.1,1,0.1)])

def test_save_square_high():
    def create_geom(**kwargs):
        from pygeoiga.nurb.cad import make_surface_square
        U, V, B = make_surface_square()
        knots = [U, V]
        from pygeoiga.nurb.refinement import knot_insertion
        to_ins = np.arange(0.1, 1, 0.1)

        knots_ins_0 = [x for x in to_ins if x not in knots[0]]
        B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_0, direction=0)
        knots_ins_1 = [x for x in to_ins if x not in knots[1]]
        B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_1, direction=1)

        geometry = dict()
        geometry["quadrat"] = {"B": B,
                               "knots": knots,
                               "kappa": 4,
                               'color': "red",
                               "position": (1, 1),
                               "BC": {0: "bot_bc", 2: "top_bc"}}
        return geometry
    size = 0.01
    name = "solution_square"
    T_t = 10
    T_b = 25
    create = True

    if create:
        create_high_resolution_answers(create_geom, T_t, T_b, name, size)

    from pygeoiga.FE_solvers.run_fenics import read_fenics_solution
    u, mesh, dofs = read_fenics_solution(datapath + name + ".h5")
    print(u(0,2,0), dofs)  # dofs = 713526

def test_compare_all_meshes_square():
    def create_geom(**kwargs):
        from pygeoiga.nurb.cad import make_surface_square
        U,V, B = make_surface_square()
        knots = [U,V]
        from pygeoiga.nurb.refinement import knot_insertion

        to_ins = np.arange(0.1, 1, 0.1)

        knots_ins_0 = [x for x in to_ins if x not in knots[0]]
        B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_0, direction=0)
        knots_ins_1 = [x for x in to_ins if x not in knots[1]]
        B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_1, direction=1)

        geometry = dict()
        geometry["quadrat"] = {"B": B,
                               "knots": knots,
                               "kappa": 4,
                               'color': "red",
                               "position": (1, 1),
                               "BC": {0: "bot_bc", 2: "top_bc"}}
        return geometry

    size = 5
    comparison_all_meshes(create_geom,
                          T_t=10,
                          T_b=25,
                          filepath = datapath + "solution_square.h5",
                          size=size,
                          knot_ins=[np.arange(0.1,1,0.1), np.arange(0.1,1,0.1)])

def test_plot_solution_biquadratic():
    def create_geom(**kwargs):
        from pygeoiga.nurb.cad import make_surface_biquadratic
        knots, B = make_surface_biquadratic()

        from pygeoiga.nurb.refinement import knot_insertion
        to_ins = kwargs.get("knot_ins", np.arange(0.1, 1, 0.1))

        knots_ins_0 = [x for x in to_ins if x not in knots[0]]
        B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_0, direction=0)
        knots_ins_1 = [x for x in to_ins if x not in knots[1]]
        B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_1, direction=1)

        geometry = dict()
        geometry["quadrat"] = {"B": B,
                               "knots": knots,
                               "kappa": 4,
                               'color': "gray",
                               "position": (1, 1),
                               "BC": {0: "bot_bc", 2: "top_bc"}}
        return geometry

    save = True
    T_t = 10
    T_b = 25
    T_l = None
    T_r = None
    name = "biquadratic"
    lith = ["4 [W/mK]"]
    levels = [12, 14, 17, 20, 22, 24]

    workflow(create_geom,
             knot_ins=np.arange(0.25, 1, 0.25),
             size=0.9,
             name=name,
             T_t=T_t,
             T_b=T_b,
             T_l=T_l,
             T_r=T_r,
             save=save,
             levels=levels,
             lith=lith,
             original=False)
    comparison_all_meshes(create_geom,
                          T_t=T_t,
                          T_b=T_b,
                          filepath=datapath + "solution_biquadratic",
                          size=0.9,
                          knot_ins=np.arange(0.25, 1, 0.25),
                          bezier=False,
                          save=save,
                          name=name+"_error")
    workflow(create_geom,
             knot_ins=np.arange(0.1, 1, 0.1),
             size=0.4,
             name=name + "_refined",
             T_t=T_t,
             T_b=T_b,
             T_l=T_l,
             T_r=T_r,
             save=save,
             levels=levels,
             original=False)
    comparison_all_meshes(create_geom,
                          T_t=T_t,
                          T_b=T_b,
                          filepath=datapath + "solution_biquadratic",
                          size=0.4,
                          knot_ins=np.arange(0.1, 1, 0.1),
                          bezier=False,
                          save=save,
                          name=name+"_refined_error")

    workflow(create_geom,
             knot_ins=np.arange(0.05, 1, 0.05),
             size=0.2,
             name=name + "_refined2",
             T_t=T_t,
             T_b=T_b,
             T_l=T_l,
             T_r=T_r,
             save=save,
             levels=levels,
             original=False)
    comparison_all_meshes(create_geom,
                          T_t=T_t,
                          T_b=T_b,
                          filepath=datapath + "solution_biquadratic",
                          size=0.2,
                          knot_ins=np.arange(0.05, 1, 0.05),
                          bezier=False,
                          save=save,
                          name=name+"_refined2_error")

def test_adaptability_mesh():
    def create_geom(knots,B, name, save):
        T_t = 10
        T_b = 25
        levels = [12, 14, 17, 20, 24]

        geometry_o = dict()
        geometry_o["quadrat"] = {"B": B,
                                 "knots": knots,
                                 "kappa": 4,
                                 'color': "gray",
                                 "position": (1, 1),
                                 "BC": {0: "bot_bc", 2: "top_bc"}}

        from pygeoiga.nurb.refinement import knot_insertion
        to_ins = np.arange(0.1, 1, 0.1)
        knots_ins_0 = [x for x in to_ins if x not in knots[0]]
        B, knots = knot_insertion(B.copy(), degree=(2, 2), knots=knots.copy(), knots_ins=knots_ins_0, direction=0)
        knots_ins_1 = [x for x in to_ins if x not in knots[1]]
        B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_1, direction=1)

        geometry = dict()
        geometry["quadrat"] = {"B": B,
                               "knots": knots,
                               "kappa": 4,
                               'color': "gray",
                               "position": (1, 1),
                               "BC": {0: "bot_bc", 2: "top_bc"}}
        geometry, gDoF = patch_topology(geometry)
        K_glob = np.zeros((gDoF, gDoF))
        K_glob = form_k_IGA_mp(geometry, K_glob)

        a = np.zeros(gDoF)
        F = np.zeros(gDoF)

        bc, a = boundary_condition_mp(geometry, a, T_t, T_b, None,None)
        bc["gDOF"] = gDoF
        a, F = solve(bc, K_glob, F, a)
        geometry = map_MP_elements(geometry, a)
        fig_sol, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(16,5), sharey=True)

        xmin = 0
        xmax = 0
        ymin = 0
        ymax = 0
        cbar = True
        for patch_id in geometry.keys():

            ax1 = p_surface(geometry_o[patch_id].get("knots"),
                            geometry_o[patch_id].get("B"),
                            border=False,
                            fill=True,
                            dim=2,
                            color = geometry_o[patch_id].get("color"),
                            ax=ax1)
            ax1 = p_cpoints(geometry_o[patch_id].get("B"),
                            ax=ax1,
                            point=True,
                            line=True,
                            color="red",
                            dim=2
                            )

            x = geometry[patch_id].get("x_sol")
            y = geometry[patch_id].get("y_sol")
            t = geometry[patch_id].get("t_sol")

            xmin = x.min() if x.min() < xmin else xmin
            xmax = x.max() if x.max() > xmax else xmax
            ymin = y.min() if y.min() < ymin else ymin
            ymax = y.max() if y.max() > ymax else ymax

            ax2 = p_temperature(x, y, t, vmin=np.min(a), vmax=np.max(a), show=False, colorbar=False,
                                ax=ax2, point=True, fill=False,
                                markersize=25)

            ax2 = p_knots(geometry[patch_id].get("knots"),
                          geometry[patch_id].get("B"),
                          ax=ax2,
                          color='k',
                          dim=2,
                          point=False,
                          line=True,
                          linestyle="--",
                          linewidth=0.2)

            ax3 = p_temperature(x, y, t, vmin=np.min(a), vmax=np.max(a), levels=200, show=False, colorbar=cbar, ax=ax3,
                                point=False, fill=True, contour=False)
            cbar = False

        ax1.set_ylabel(r"$y$")
        for ax in ax1, ax2, ax3:
            ax.set_aspect("equal")
            ax.set_xlabel(r"$x$")
            ax.set_xlim(xmin-0.1, xmax+0.1)
            ax.set_ylim(ymin-0.1, ymax+0.1)

        for c in ax3.collections:
            c.set_edgecolor("face")

        for patch_id in geometry.keys():
            x = geometry[patch_id].get("x_sol")
            y = geometry[patch_id].get("y_sol")
            t = geometry[patch_id].get("t_sol")

            ax3 = p_temperature(x, y, t, vmin=np.min(a), vmax=np.max(a), levels=levels,
                                show=False, colorbar=False, colors=["black"], ax=ax3,
                                point=False, fill=False, contour=True, cmap=None,
                                linewidths=0.5)
        # ax3 = p_temperature_mp(geometry, vmin=np.min(a), vmax=np.max(a), levels=levels,
        #                                         show=False, colorbar=False, colors=["black"], ax=ax3,
        #                                            point=False, fill=False, contour=True, cmap=None,
        #                                            linewidths=0.5)

        ax1.set_title("Coarse control net and surface")
        ax2.set_title("NURBS-mesh after refinement")
        ax3.set_title("Temperature field")
        plt.tight_layout()
        fig_sol.show()

        if save or save_all:
            fig_sol.savefig(fig_folder + name +".pdf", **kwargs_savefig)

    from pygeoiga.nurb.cad import make_surface_biquadratic
    knots, B = make_surface_biquadratic()
    save = True
    create_geom(knots, B, save=save, name="original")
    B[-1,-1] = np.array([6,6,1])
    create_geom(knots, B, save=save, name="adapted")

def test_error_plot_convergence():
    from pygeoiga.FE_solvers.run_fenics import read_fenics_solution
    u, mesh, dofs = read_fenics_solution(datapath + "solution_biquadratic")
    u.set_allow_extrapolation(True)

    def create_geom(**kwargs):
        from pygeoiga.nurb.cad import make_surface_biquadratic
        knots, B = make_surface_biquadratic()

        from pygeoiga.nurb.refinement import knot_insertion
        to_ins = kwargs.get("knot_ins", np.arange(0.1, 1, 0.1))

        knots_ins_0 = [x for x in to_ins if x not in knots[0]]
        B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_0, direction=0)
        knots_ins_1 = [x for x in to_ins if x not in knots[1]]
        B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_1, direction=1)

        geometry = dict()
        geometry["quadrat"] = {"B": B,
                               "knots": knots,
                               "kappa": 4,
                               'color': "gray",
                               "position": (1, 1),
                               "BC": {0: "bot_bc", 2: "top_bc"}}
        return geometry

    def calculate_error_IGA(geometry):
        diff_all = np.array([])
        for patch_id in geometry.keys():
            x = geometry[patch_id].get("x_sol")
            y = geometry[patch_id].get("y_sol")
            m, n = x.shape
            correct = np.zeros((m, n))
            for x_i in range(n):
                for y_i in range(m):
                    correct[y_i, x_i] = u(x[y_i, x_i], y[y_i, x_i], 0)
            #err = correct - geometry[patch_id].get("t_sol")
            err = (correct - geometry[patch_id].get("t_sol"))**2
            diff_all = np.r_[diff_all, err.ravel()]
        l2 = np.sqrt(np.sum(diff_all))
        return l2

    def calculate_error_fem(geometry, u_FEM, u):
        diff_all = np.array([])
        for patch_id in geometry.keys():
            x = geometry[patch_id].get("x_sol")
            y = geometry[patch_id].get("y_sol")
            m, n = x.shape
            correct = np.zeros((m, n))
            correct_FEM = np.zeros((m, n))
            for x_i in range(n):
                for y_i in range(m):
                    correct_FEM[y_i, x_i] = u_FEM(x[y_i, x_i], y[y_i, x_i], 0)
                    correct[y_i, x_i] = u(x[y_i, x_i], y[y_i, x_i], 0)
            #erro = correct - correct_FEM
            erro = (correct-correct_FEM)**2
            diff_all = np.r_[diff_all, erro.ravel()]
        l2 = np.sqrt(np.sum(diff_all))
        return l2

    T_t = 10
    T_b = 25

    knot_ins_list = [np.arange(0.25, 1, 0.25),
                     np.arange(0.1, 1, 0.1),
                     np.arange(0.05, 1, 0.05),
                     np.arange(0.01, 1, 0.01)
                     ]
    size_list = [0.9, 0.4, 0.2, 0.1]
    e_IGA = np.array([])
    d_IGA = np.array([])
    e_FEM = np.array([])
    d_FEM = np.array([])
    for points in range(len(knot_ins_list)):
        knot_ins = knot_ins_list[points]
        size = size_list[points]
        geometry_IGA, dof_IGA = do_IGA(create_geom, T_t, T_b, knot_ins)
        nodal_coordinates, temperature_nodes, u_FEM = do_FEM(create_geom, T_t, T_b, knot_ins, size)
        dof_FEM = temperature_nodes.shape[0]

        l2_IGA = calculate_error_IGA(geometry_IGA)
        l2_FEM = calculate_error_fem(geometry_IGA, u_FEM, u)

        e_IGA = np.append(e_IGA,l2_IGA)
        d_IGA = np.append(d_IGA, dof_IGA)
        e_FEM = np.append(e_FEM, l2_FEM)
        d_FEM = np.append(d_FEM, dof_FEM)

    fig, ax = plt.subplots()

    ax.plot(d_IGA, e_IGA, 'r-*' )
    ax.plot(d_FEM, e_FEM, 'b-*')

    fig.show()



