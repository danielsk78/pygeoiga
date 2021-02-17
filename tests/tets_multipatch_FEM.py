import numpy as np
import matplotlib.pyplot as plt
from pygeoiga.analysis.MultiPatch import patch_topology, bezier_extraction_mp, form_k_bezier_mp, map_MP_elements
from pygeoiga.analysis.MultiPatch import boundary_condition_mp
from pygeoiga.analysis.common import solve

import pygeoiga as gn
from pygeoiga.plot.nrbplotting_mpl import p_surface, p_cpoints, p_knots, create_figure
from pygeoiga.plot.solution_mpl import p_temperature, p_temperature_mp


def test_multi_patch_bezier_extraction():
    from pygeoiga.nurb.cad import make_3_layer_patches

    geometry = make_3_layer_patches(refine=True)
    geometry, gDoF = patch_topology(geometry)
    geometry = bezier_extraction_mp(geometry)

def test_form_k():
    from pygeoiga.nurb.cad import make_3_layer_patches

    geometry = make_3_layer_patches(refine=True)
    geometry, gDoF = patch_topology(geometry)
    geometry = bezier_extraction_mp(geometry)

    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_bezier_mp(geometry, K_glob)

    plt.spy(K_glob)
    plt.show()


def plot_mp_FEM(geometry, a, gDoF, levels=None):
    fig_sol, [ax2, ax3] = plt.subplots(1, 2, figsize=(10,5), sharey=True)
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

    for ax in ax2, ax3:
        ax.set_aspect("equal")
        ax.set_ylabel(r"$y$")
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

    plt.tight_layout()
    fig_sol.show()

def test_3_layers():
    from pygeoiga.nurb.cad import make_3_layer_patches

    geometry = make_3_layer_patches(refine=True)
    geometry, gDoF = patch_topology(geometry)
    geometry = bezier_extraction_mp(geometry)

    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_bezier_mp(geometry, K_glob)

    D = np.zeros(gDoF)
    b = np.zeros(gDoF)

    T_t = 10  # [°C]
    T_b = 25  # [°C]
    T_l = None  # 10
    T_r = None  # 40
    bc, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    D, b = solve(bc, K_glob, b, D)

    geometry = map_MP_elements(geometry, D)
    plot_mp_FEM(geometry, D, gDoF, levels = [12, 14, 17, 20, 22, 24])

def test_plot_solution_fault_model_mp():
    from pygeoiga.nurb.cad import make_fault_model
    levels = [12, 16, 20, 24, 28, 32, 36]
    T_t = 10
    T_b = 40
    T_l = None
    T_r = None

    geometry = make_fault_model(refine=True)
    geometry, gDoF = patch_topology(geometry)
    geometry = bezier_extraction_mp(geometry)

    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_bezier_mp(geometry, K_glob)

    D = np.zeros(gDoF)
    b = np.zeros(gDoF)

    bc, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    D, b = solve(bc, K_glob, b, D)

    geometry = map_MP_elements(geometry, D)
    plot_mp_FEM(geometry, D, gDoF, levels = levels)


def test_plot_solution_salt_dome_mp():

    from pygeoiga.nurb.cad import make_salt_dome
    levels = [15, 20, 30, 40, 50, 60, 70, 80, 85]
    T_t = 10
    T_b = 90
    T_l = None
    T_r = None

    geometry = make_salt_dome(refine=True)
    geometry, gDoF = patch_topology(geometry)
    geometry = bezier_extraction_mp(geometry)

    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_bezier_mp(geometry, K_glob)

    D = np.zeros(gDoF)
    b = np.zeros(gDoF)

    bc, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    D, b = solve(bc, K_glob, b, D)

    geometry = map_MP_elements(geometry, D)
    plot_mp_FEM(geometry, D, gDoF, levels = levels)

def test_comparison_efficiency():
    from time import process_time
    levels = [15, 20, 30, 40, 50, 60, 70, 80, 85]
    T_t = 10
    T_b = 90
    T_l = None
    T_r = None
    from pygeoiga.nurb.cad import make_salt_dome

    start_model = process_time()
    geometry = make_salt_dome(refine=True)
    finish_model = process_time()

    start_FEM = process_time()
    geometry, gDoF = patch_topology(geometry)
    geometry = bezier_extraction_mp(geometry)

    K_glob = np.zeros((gDoF, gDoF))
    s_k_FEM = process_time()
    K_glob = form_k_bezier_mp(geometry, K_glob)
    e_k_FEM = process_time()

    D = np.zeros(gDoF)
    b = np.zeros(gDoF)

    bc, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    D, b = solve(bc, K_glob, b, D)
    finish_FEM = process_time()

    from pygeoiga.analysis.MultiPatch import form_k_IGA_mp

    geometry = make_salt_dome(refine=True)  # refine=np.arange(0.05,1,0.05))

    start_IGA = process_time()
    geometry, gDoF = patch_topology(geometry)
    K_glob = np.zeros((gDoF, gDoF))
    s_k_IGA = process_time()
    K_glob = form_k_IGA_mp(geometry, K_glob)
    e_k_IGA= process_time()
    D = np.zeros(gDoF)
    b = np.zeros(gDoF)

    T_t = 10  # [°C]
    T_b = 90  # [°C]
    T_l = None
    T_r = None
    bc, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    D, b = solve(bc, K_glob, b, D)
    finish_IGA = process_time()

    time_FEM = finish_FEM - start_FEM
    time_k_fem = e_k_FEM - s_k_FEM
    time_IGA = finish_IGA - start_IGA
    time_k_iga = e_k_IGA - s_k_IGA
    time_model_refinement = finish_model - start_model

    print("gDoF: ", gDoF)
    print("FEM: ", time_FEM)  # FEM:  241.05022452400001
    print("K_FEM; ", time_k_fem)  # K_FEM;  172.537146357
    print("IGA: ", time_IGA)  # IGA: 372.34105559899996
    print("K_IGA; ", time_k_iga)  # K_IGA;  316.763859975
    print("Refinement: ", time_model_refinement)  # Refinement:  0.06600132199999997